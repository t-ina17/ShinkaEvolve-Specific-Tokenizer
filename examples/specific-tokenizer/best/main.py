# EVOLVE-BLOCK-START
"""Hybrid TF‑IDF / TF‑Jaccard matcher for JP/EN e‑commerce search.

Public API (unchanged):
- tokenize(text) -> list[str]
- score_match(query, product) -> float
- run_experiment(examples, ndcg_k=10) -> dict
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# 1️⃣ Normalisation & synonym handling
# ----------------------------------------------------------------------
_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+")

# synonym patterns – compiled once
_SYNONYM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bk\s*([0-9]+)\b", flags=re.IGNORECASE), r"\1金"),
    (re.compile(r"\b([0-9]+)k\b", flags=re.IGNORECASE), r"\1金"),
    (re.compile(r"\busb[-\s]?c\b", flags=re.IGNORECASE), "usb c"),
    (re.compile(r"\btype[-\s]?c\b", flags=re.IGNORECASE), "usb c"),
]


def _apply_synonyms(s: str) -> str:
    for pat, repl in _SYNONYM_PATTERNS:
        s = pat.sub(repl, s)
    return s


def normalize_text(text: str) -> str:
    """NFKC → lower → synonym → collapse spaces."""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _apply_synonyms(s)
    s = _RE_SPACES.sub(" ", s).strip()
    return s


# ----------------------------------------------------------------------
# 2️⃣ Tokenisation (word tokens + char‑2/3‑grams)
# ----------------------------------------------------------------------
def _char_ngrams(token: str, n: int) -> List[str]:
    """Return character n‑grams of a single token (n ≥ 2)."""
    if len(token) < n:
        return []
    return [token[i : i + n] for i in range(len(token) - n + 1)]


def tokenize(text: str) -> List[str]:
    """
    - Normalise
    - Split on non‑alnum/Japanese characters
    - Keep tokens of length ≥2
    - Append 2‑gram and 3‑gram character n‑grams for each token
    """
    s = normalize_text(text)
    if not s:
        return []
    raw = [p for p in _RE_TOKEN_SPLIT.split(s) if p]
    base = [p for p in raw if len(p) >= 2]

    ngrams: List[str] = []
    for token in base:
        ngrams.extend(_char_ngrams(token, 2))
        ngrams.extend(_char_ngrams(token, 3))

    return base + ngrams


# ----------------------------------------------------------------------
# 3️⃣ TF utilities
# ----------------------------------------------------------------------
def _tf(tokens: Iterable[str]) -> Dict[str, int]:
    """Term‑frequency dictionary."""
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


# ----------------------------------------------------------------------
# 4️⃣ Global IDF model (built per batch)
# ----------------------------------------------------------------------
_IDF_CACHE: Dict[str, float] = {}
_TOTAL_DOCS: int = 0  # number of product documents in the current batch


def _build_idf(product_texts: Iterable[str]) -> None:
    """Populate _IDF_CACHE from all product titles in the batch."""
    global _IDF_CACHE, _TOTAL_DOCS
    df: Dict[str, int] = defaultdict(int)

    for txt in product_texts:
        tokens = set(tokenize(txt))
        for t in tokens:
            df[t] += 1

    # count documents (may be less than len(product_texts) if some are empty)
    _TOTAL_DOCS = max(1, sum(1 for _ in product_texts))

    _IDF_CACHE = {}
    for token, freq in df.items():
        # smooth IDF, add 1 to keep weights >0
        _IDF_CACHE[token] = math.log((_TOTAL_DOCS + 1) / (freq + 1)) + 1.0


def _idf(token: str) -> float:
    """Return IDF weight; unseen tokens get a neutral weight of 1.0."""
    return _IDF_CACHE.get(token, 1.0)


def _tfidf_vector(tf: Dict[str, int]) -> Tuple[Dict[str, float], float]:
    """Convert TF dict → (sparse TF‑IDF vector, L2 norm)."""
    vec: Dict[str, float] = {}
    sq_sum = 0.0
    for token, freq in tf.items():
        w = freq * _idf(token)
        vec[token] = w
        sq_sum += w * w
    return vec, math.sqrt(sq_sum)


def _cosine_similarity(
    vec_a: Dict[str, float],
    norm_a: float,
    vec_b: Dict[str, float],
    norm_b: float,
) -> float:
    """Sparse cosine similarity."""
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # iterate over the smaller dict for speed
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a
    dot = 0.0
    for token, w_a in vec_a.items():
        w_b = vec_b.get(token)
        if w_b is not None:
            dot += w_a * w_b
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# 5️⃣ TF‑Jaccard (bag‑of‑words) – unchanged core logic
# ----------------------------------------------------------------------
def _tf_jaccard(q_tokens: List[str], p_tokens: List[str]) -> float:
    if not q_tokens or not p_tokens:
        return 0.0
    q_tf = _tf(q_tokens)
    p_tf = _tf(p_tokens)

    inter = 0.0
    union = 0.0
    for tok in set(q_tf) | set(p_tf):
        inter += min(q_tf.get(tok, 0), p_tf.get(tok, 0))
        union += max(q_tf.get(tok, 0), p_tf.get(tok, 0))
    if union == 0:
        return 0.0
    return inter / union


# ----------------------------------------------------------------------
# 6️⃣ Hybrid scoring (cosine + Jaccard) + heuristics
# ----------------------------------------------------------------------
_ALPHA = 0.7  # weight for cosine; (1‑α) for Jaccard


def _hybrid_score(
    query: str,
    product: str,
    q_vec: Dict[str, float],
    q_norm: float,
    p_vec: Dict[str, float],
    p_norm: float,
    q_tokens: List[str],
    p_tokens: List[str],
) -> float:
    # 1️⃣ cosine similarity (TF‑IDF)
    cos = _cosine_similarity(q_vec, q_norm, p_vec, p_norm)

    # 2️⃣ TF‑Jaccard
    jac = _tf_jaccard(q_tokens, p_tokens)

    # 3️⃣ blend
    score = _ALPHA * cos + (1 - _ALPHA) * jac

    # 4️⃣ short‑query sqrt scaling (same as baseline)
    if len(q_tokens) <= 3:
        score = math.sqrt(score)

    # 5️⃣ exact‑coverage boost
    if set(q_tokens).issubset(set(p_tokens)):
        score = min(1.0, score * 1.5)

    # 6️⃣ any shared token boost (capped)
    if set(q_tokens) & set(p_tokens):
        score = min(1.0, score * 1.2)

    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# 7️⃣ Public scoring API (with caching for products)
# ----------------------------------------------------------------------
# Caches populated in run_experiment – they are cleared each call.
_PRODUCT_TOKENS: Dict[int, List[str]] = {}
_PRODUCT_VECTORS: Dict[int, Tuple[Dict[str, float], float]] = {}


def score_match(query: str, product: str) -> float:
    """Public entry – computes hybrid similarity."""
    # In a single‑pair call (outside run_experiment) fall back to baseline TF‑Jaccard.
    if not _IDF_CACHE:
        return _tf_jaccard(tokenize(query), tokenize(product))

    # Compute query side on‑the‑fly
    q_tokens = tokenize(query)
    q_tf = _tf(q_tokens)
    q_vec, q_norm = _tfidf_vector(q_tf)

    # Retrieve cached product representation (if available)
    prod_id = id(product)  # cheap deterministic key for the current run
    if prod_id not in _PRODUCT_TOKENS:
        p_tokens = tokenize(product)
        p_tf = _tf(p_tokens)
        p_vec, p_norm = _tfidf_vector(p_tf)
        _PRODUCT_TOKENS[prod_id] = p_tokens
        _PRODUCT_VECTORS[prod_id] = (p_vec, p_norm)
    else:
        p_tokens = _PRODUCT_TOKENS[prod_id]
        p_vec, p_norm = _PRODUCT_VECTORS[prod_id]

    return _hybrid_score(
        query, product, q_vec, q_norm, p_vec, p_norm, q_tokens, p_tokens
    )


def score_match_with_meta(query: str, product: str) -> Tuple[float, dict]:
    """Same as score_match but returns token lists for debugging."""
    q_tokens = tokenize(query)
    q_tf = _tf(q_tokens)
    q_vec, q_norm = _tfidf_vector(q_tf)

    prod_id = id(product)
    if prod_id not in _PRODUCT_TOKENS:
        p_tokens = tokenize(product)
        p_tf = _tf(p_tokens)
        p_vec, p_norm = _tfidf_vector(p_tf)
        _PRODUCT_TOKENS[prod_id] = p_tokens
        _PRODUCT_VECTORS[prod_id] = (p_vec, p_norm)
    else:
        p_tokens = _PRODUCT_TOKENS[prod_id]
        p_vec, p_norm = _PRODUCT_VECTORS[prod_id]

    score = _hybrid_score(
        query, product, q_vec, q_norm, p_vec, p_norm, q_tokens, p_tokens
    )

    meta = {"q_tokens": q_tokens[:20], "p_tokens": p_tokens[:20]}
    return float(score), meta


# ----------------------------------------------------------------------
# 8️⃣ Experiment runner (builds IDF & product caches once)
# ----------------------------------------------------------------------
def run_experiment(examples: List[dict], ndcg_k: int = 10) -> dict:
    """Entry point used by the evaluator."""
    # ------------------------------------------------------------------
    # 1️⃣ Build global IDF from all product titles in this batch
    # ------------------------------------------------------------------
    product_texts = [ex.get("product", "") for ex in examples]
    _build_idf(product_texts)

    # ------------------------------------------------------------------
    # 2️⃣ Pre‑compute product vectors & token lists (cached globally)
    # ------------------------------------------------------------------
    _PRODUCT_TOKENS.clear()
    _PRODUCT_VECTORS.clear()
    for txt in product_texts:
        pid = id(txt)
        if pid in _PRODUCT_TOKENS:
            continue
        tokens = tokenize(txt)
        tf = _tf(tokens)
        vec, norm = _tfidf_vector(tf)
        _PRODUCT_TOKENS[pid] = tokens
        _PRODUCT_VECTORS[pid] = (vec, norm)

    # ------------------------------------------------------------------
    # 3️⃣ Score each (query, product) pair
    # ------------------------------------------------------------------
    scored: List[dict] = []
    for ex in examples:
        qid = ex.get("query_id", "")
        query = ex.get("query", "")
        product = ex.get("product", "")
        label = ex.get("label", None)
        score = score_match(query, product)
        scored.append(
            {
                "query_id": qid,
                "query": query,
                "product": product,
                "label": label,
                "score": float(score),
            }
        )

    # ------------------------------------------------------------------
    # 4️⃣ Aggregate – unsupervised mean or NDCG@k
    # ------------------------------------------------------------------
    has_labels = any(r.get("label") is not None for r in scored)
    if not has_labels:
        mean_score = sum(r["score"] for r in scored) / max(1, len(scored))
        return {
            "mode": "unsupervised_mean",
            "combined_score": float(mean_score),
            "num_rows": len(scored),
            "preview": scored[:5],
        }

    # group by query_id
    by_qid: Dict[str, List[dict]] = {}
    for r in scored:
        by_qid.setdefault(str(r.get("query_id", "")), []).append(r)

    ndcgs: List[float] = []
    for _qid, rows in by_qid.items():
        rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
        labels = [float(x.get("label") or 0.0) for x in rows_sorted]
        ndcgs.append(_ndcg(labels, k=int(ndcg_k)))

    mean_ndcg = sum(ndcgs) / max(1, len(ndcgs))
    return {
        "mode": "labeled_ndcg",
        "combined_score": float(mean_ndcg),
        "num_queries": len(by_qid),
        "num_rows": len(scored),
        "ndcg_k": int(ndcg_k),
        "preview": scored[:5],
    }


# ----------------------------------------------------------------------
# 9️⃣ NDCG helpers (unchanged)
# ----------------------------------------------------------------------
def _dcg(relevances: List[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        s += (2.0 ** float(rel) - 1.0) / math.log2(i + 2)
    return s


def _ndcg(relevances: List[float], k: int) -> float:
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return float(dcg / ideal)


# EVOLVE-BLOCK-END
