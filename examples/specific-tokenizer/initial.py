# EVOLVE-BLOCK-START
"""Domain-specific tokenizer + query/product match scorer.

このブロックが主に進化対象です。
- tokenize(text) -> list[str]
- score_match(query, product) -> float

制約:
- 例外を投げずに安定して動くこと
- 実行時間が過度に増えないこと
"""

from __future__ import annotations

import re
import math
import unicodedata
from typing import Iterable, List


_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN_SPLIT = re.compile(
    r"[^0-9A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+"
)

# Sudachi 品詞フィルタ（形態素解析が使える場合のみ適用）
# 「動詞/形容詞も入れると良いか？」を探索できるように、複数パターンを候補として持つ。
# - Sudachi の品詞は (品詞大分類, 品詞中分類, ... ) のタプル
# - ここでは大分類だけで粗くフィルタ
# - score_match 側で候補を試して、最もスコアが良い品詞セットを採用する
SUDACHI_POS_CANDIDATE_ALLOWLISTS = [
    {"名詞"},
    {"名詞", "動詞"},
    {"名詞", "形容詞"},
    {"名詞", "動詞", "形容詞"},
]


def _allowlist_label(allow: set[str]) -> str:
    if not allow:
        return "ALL"
    return "+".join(sorted(allow))


def _contains_japanese(text: str) -> bool:
    return any("\u3040" <= ch <= "\u9FFF" for ch in text)


_SUDACHI_TOKENIZER = None


def _get_sudachi_tokenizer():
    """Lazy-load Sudachi tokenizer if installed; otherwise return None."""
    global _SUDACHI_TOKENIZER
    if _SUDACHI_TOKENIZER is not None:
        return _SUDACHI_TOKENIZER
    try:
        from sudachipy import dictionary  # type: ignore
    except Exception:
        _SUDACHI_TOKENIZER = None
        return None

    try:
        _SUDACHI_TOKENIZER = dictionary.Dictionary().create()
    except Exception:
        _SUDACHI_TOKENIZER = None
    return _SUDACHI_TOKENIZER


def _sudachi_morph_pairs(text: str) -> List[tuple[str, str]]:
    """SudachiPy で形態素解析し、(token, pos1) の列を返す（失敗時は空）."""
    tok = _get_sudachi_tokenizer()
    if tok is None:
        return []

    try:
        from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore

        mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        morphemes = tok.tokenize(text, mode)
    except Exception:
        return []

    out: List[tuple[str, str]] = []
    for m in morphemes:
        try:
            s = m.normalized_form()
        except Exception:
            try:
                s = m.surface()
            except Exception:
                s = ""
        s = normalize_text(s)
        if not s:
            continue

        # 品詞取得（失敗したら空扱い）
        try:
            pos = m.part_of_speech()
            pos1 = str(pos[0]) if pos and len(pos) > 0 else ""
        except Exception:
            pos1 = ""

        # 1文字はノイズになりやすいので落とす（例: 「の」「に」など）
        if len(s) < 2:
            continue

        out.append((s, pos1))
    return out


def _sudachi_tokens_from_pairs(
    pairs: List[tuple[str, str]], allow_pos1: set[str]
) -> List[str]:
    if not pairs:
        return []
    if not allow_pos1:
        return [t for (t, _) in pairs]
    return [t for (t, p1) in pairs if (p1 and p1 in allow_pos1)]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def _basic_tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = [p for p in _RE_TOKEN_SPLIT.split(text) if p]
    return parts


def _char_ngrams(token: str, n: int) -> List[str]:
    if n <= 0:
        return []
    if len(token) <= n:
        return [token]
    return [token[i : i + n] for i in range(0, len(token) - n + 1)]


def tokenize(text: str) -> List[str]:
    """形態素解析（SudachiPy）+ フォールバック。

    形態素解析器を使う場合は、ここを発展させてください。
    """
    base = _basic_tokenize(text)

    # tokenize() は単一のデフォルト挙動として「名詞のみ」を採用。
    # どの品詞が良いかは score_match 側で候補探索する。
    sudachi_tokens: List[str] = []
    if _contains_japanese(text):
        pairs = _sudachi_morph_pairs(text)
        sudachi_tokens = _sudachi_tokens_from_pairs(pairs, {"名詞"})

    # Sudachi が取れた場合はそれを主要トークンにし、取りこぼし対策で base も混ぜる。
    base_tokens = sudachi_tokens if sudachi_tokens else base

    out: List[str] = []
    for tok in base_tokens:
        if not tok:
            continue
        out.append(tok)
        # 日本語っぽいトークンは2-gram/3-gramで分割も追加
        has_jp = _contains_japanese(tok)
        if has_jp and len(tok) >= 3:
            out.extend(_char_ngrams(tok, 2))
            out.extend(_char_ngrams(tok, 3))

        # 英数・型番の部分一致も拾えるように、長めのトークンはn-gramを追加
        if len(tok) >= 4:
            out.extend(_char_ngrams(tok, 3))
            out.extend(_char_ngrams(tok, 4))

    # base を混ぜて英数・記号区切りの取りこぼしを減らす
    if sudachi_tokens:
        out.extend([t for t in base if t])

    # 低情報トークン除去（最小限）
    out = [t for t in out if len(t) >= 2]
    return out


def _tf(tokens: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


def score_match(query: str, product: str) -> float:
    """query と product のマッチ度スコア（大きいほど良い）。

    現状は軽量な重み付きJaccardに近いスコア。
    """
    def _score_from_tokens(q_toks: List[str], p_toks: List[str]) -> float:
        if not q_toks or not p_toks:
            return 0.0

        q_tf = _tf(q_toks)
        p_tf = _tf(p_toks)

        inter = 0.0
        union = 0.0
        all_keys = set(q_tf) | set(p_tf)
        for k in all_keys:
            a = q_tf.get(k, 0)
            b = p_tf.get(k, 0)
            inter += float(min(a, b))
            union += float(max(a, b))

        if union <= 0:
            score = 0.0
        else:
            score = inter / union

        # クエリが短い場合に、完全一致に寄せる軽い補正
        if len(q_toks) <= 3:
            score = math.sqrt(score)

        if not math.isfinite(score):
            return 0.0
        return float(max(0.0, score))

    score, _meta = score_match_with_meta(query, product)
    return float(score)


def score_match_with_meta(query: str, product: str) -> tuple[float, dict]:
    """score_match と同じスコアを返しつつ、採用した品詞候補などのメタ情報も返す。"""

    def _score_from_tokens(q_toks: List[str], p_toks: List[str]) -> float:
        if not q_toks or not p_toks:
            return 0.0

        q_tf = _tf(q_toks)
        p_tf = _tf(p_toks)

        inter = 0.0
        union = 0.0
        all_keys = set(q_tf) | set(p_tf)
        for k in all_keys:
            a = q_tf.get(k, 0)
            b = p_tf.get(k, 0)
            inter += float(min(a, b))
            union += float(max(a, b))

        if union <= 0:
            score = 0.0
        else:
            score = inter / union

        if len(q_toks) <= 3:
            score = math.sqrt(score)

        if not math.isfinite(score):
            return 0.0
        return float(max(0.0, score))

    # ベースライン（tokenize() はデフォルトのSudachi名詞＋フォールバック混ぜ）
    best_score = _score_from_tokens(tokenize(query), tokenize(product))
    best_allow_label = "fallback"
    used_sudachi = False

    # Sudachi が使える場合は、品詞allowlist候補を試して一番良いものを採用
    if _contains_japanese(query) or _contains_japanese(product):
        q_pairs = _sudachi_morph_pairs(query) if _contains_japanese(query) else []
        p_pairs = _sudachi_morph_pairs(product) if _contains_japanese(product) else []

        # Sudachi が片方でも取れた時だけ候補探索
        if q_pairs or p_pairs:
            used_sudachi = True
            base_q = _basic_tokenize(query)
            base_p = _basic_tokenize(product)
            for allow in SUDACHI_POS_CANDIDATE_ALLOWLISTS:
                q_sud = _sudachi_tokens_from_pairs(q_pairs, allow) if q_pairs else []
                p_sud = _sudachi_tokens_from_pairs(p_pairs, allow) if p_pairs else []
                q_toks = q_sud + base_q if q_sud else base_q
                p_toks = p_sud + base_p if p_sud else base_p
                s = _score_from_tokens(q_toks, p_toks)
                if s > best_score:
                    best_score = s
                    best_allow_label = _allowlist_label(allow)

    meta = {
        "used_sudachi": bool(used_sudachi),
        "best_pos_allowlist": best_allow_label,
    }
    return float(best_score), meta


# EVOLVE-BLOCK-END


def run_experiment(
    examples: list[dict],
    ndcg_k: int = 10,
) -> dict:
    """評価器から呼ばれる入口。

    examples: list of {query_id, query, product, label?}

    戻り値は evaluate.py 側で集約される。
    """
    # スコア計算
    scored = []
    pos_choice_counts: dict[str, int] = {}
    used_sudachi_count = 0
    for ex in examples:
        query = ex.get("query", "")
        product = ex.get("product", "")
        label = ex.get("label", None)
        qid = ex.get("query_id", "")
        s, meta = score_match_with_meta(query, product)
        if meta.get("used_sudachi"):
            used_sudachi_count += 1
        choice = str(meta.get("best_pos_allowlist") or "")
        if choice:
            pos_choice_counts[choice] = pos_choice_counts.get(choice, 0) + 1
        scored.append(
            {
                "query_id": qid,
                "query": query,
                "product": product,
                "label": label,
                "score": float(s),
            }
        )

    has_labels = any(r["label"] is not None for r in scored)

    if not has_labels:
        # 教師なし: 平均スコア
        mean_score = sum(r["score"] for r in scored) / max(1, len(scored))
        return {
            "mode": "unsupervised_mean",
            "combined_score": float(mean_score),
            "num_rows": len(scored),
            "sudachi_used_rows": int(used_sudachi_count),
            "pos_allowlist_counts": pos_choice_counts,
            "preview": scored[:5],
        }

    # ラベルあり: query_id ごとにNDCG@k
    # labelが欠ける行は0扱い
    by_qid: dict[str, list[dict]] = {}
    for r in scored:
        qid = str(r.get("query_id", ""))
        by_qid.setdefault(qid, []).append(r)

    ndcgs: list[float] = []
    for qid, rows in by_qid.items():
        rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
        labels = [float(x.get("label") or 0.0) for x in rows_sorted]
        ndcgs.append(_ndcg(labels, k=ndcg_k))

    mean_ndcg = sum(ndcgs) / max(1, len(ndcgs))

    return {
        "mode": "labeled_ndcg",
        "combined_score": float(mean_ndcg),
        "num_queries": len(by_qid),
        "num_rows": len(scored),
        "ndcg_k": int(ndcg_k),
        "sudachi_used_rows": int(used_sudachi_count),
        "pos_allowlist_counts": pos_choice_counts,
        "preview": scored[:5],
    }


def _dcg(relevances: list[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        denom = math.log2(i + 2)
        s += (2.0 ** float(rel) - 1.0) / denom
    return s


def _ndcg(relevances: list[float], k: int) -> float:
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return float(dcg / ideal)
