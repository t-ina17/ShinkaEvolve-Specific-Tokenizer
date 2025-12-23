# EVOLVE-BLOCK-START
"""Jaccard ベースのスコア（最小構成の初期個体）。

このファイルは ShinkaEvolve の「進化対象プログラム（初期個体）」です。
論文掲載・再現実験を想定し、外部依存を増やさずに動く最小の実装にしています。

提供するAPI（進化で書き換えられる想定）
- `tokenize(text) -> list[str]`
- `score_match(query, product) -> float`
- `run_experiment(examples, ndcg_k) -> dict`

評価器（examples/specific-tokenizer/evaluate.py）から `run_experiment` が呼ばれます。
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, List


Token = str
Tokens = List[Token]

_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+")


def normalize_text(text: str) -> str:
    """テキスト正規化（NFKC + lower + 連続空白の圧縮）。"""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _RE_SPACES.sub(" ", s).strip()
    return s


def tokenize(text: str) -> Tokens:
    """正規表現ベースの簡易トークナイズ。

    - 文字種の境界（記号・空白など）で分割
    - 1文字トークンはノイズになりやすいので除去

    注意: ここは最小のベースラインです。日本語形態素解析や n-gram 追加などは、
    進化（もしくは別の初期個体）で導入する前提です。
    """
    s = normalize_text(text)
    if not s:
        return []
    parts = [p for p in _RE_TOKEN_SPLIT.split(s) if p]
    return [p for p in parts if len(p) >= 2]


def _tf(tokens: Iterable[str]) -> dict[str, int]:
    """トークン列を TF（出現回数）辞書に変換する。"""
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _tf_jaccard(q_tokens: Tokens, p_tokens: Tokens) -> float:
    """TF（出現回数）を考慮した Jaccard を返す。

    集合Jaccardではなく、
    - 交差: min(tf_q, tf_p)
    - 和集合: max(tf_q, tf_p)
    を用いる "bag-of-words Jaccard"。
    """
    if not q_tokens or not p_tokens:
        return 0.0

    q_tf = _tf(q_tokens)
    p_tf = _tf(p_tokens)

    inter = 0.0
    union = 0.0
    for tok in set(q_tf) | set(p_tf):
        inter += float(min(q_tf.get(tok, 0), p_tf.get(tok, 0)))
        union += float(max(q_tf.get(tok, 0), p_tf.get(tok, 0)))

    if union <= 0.0:
        return 0.0

    score = inter / union
    if not math.isfinite(score):
        return 0.0
    return float(max(0.0, score))


def score_match(query: str, product: str) -> float:
    """query と product のマッチ度スコア（大きいほど良い）。

    ベースは TF-Jaccard。
    """
    score, _meta = score_match_with_meta(query, product)
    return float(score)


def score_match_with_meta(query: str, product: str) -> tuple[float, dict]:
    """スコアとメタ情報を返す。

    返す meta は、定性分析（どんなトークンが効いたか）やデバッグに使う。
    """
    q = str(query or "")
    p = str(product or "")

    q_tokens = tokenize(q)
    p_tokens = tokenize(p)

    score = _tf_jaccard(q_tokens, p_tokens)

    # クエリが短い場合は完全一致寄りにする軽い補正（任意）
    if len(q_tokens) <= 3:
        score = math.sqrt(score)

    if not math.isfinite(score):
        score = 0.0

    meta = {
        "q_tokens": q_tokens[:20],
        "p_tokens": p_tokens[:20],
    }
    return float(max(0.0, score)), meta


def run_experiment(examples: list[dict], ndcg_k: int = 10) -> dict:
    """評価器から呼ばれる入口。

    Parameters
    - examples: list[dict]
        各要素は {query_id, query, product, label?} を想定。
    - ndcg_k: int
        NDCG@k の k。

    Returns
    - dict: `combined_score` を必ず含む。
    """
    scored: list[dict] = []
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

    has_labels = any(r.get("label") is not None for r in scored)
    if not has_labels:
        mean_score = sum(r["score"] for r in scored) / max(1, len(scored))
        return {
            "mode": "unsupervised_mean",
            "combined_score": float(mean_score),
            "num_rows": len(scored),
            "preview": scored[:5],
        }

    by_qid: dict[str, list[dict]] = {}
    for r in scored:
        by_qid.setdefault(str(r.get("query_id", "")), []).append(r)

    ndcgs: list[float] = []
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


def _dcg(relevances: list[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        s += (2.0 ** float(rel) - 1.0) / math.log2(i + 2)
    return s


def _ndcg(relevances: list[float], k: int) -> float:
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return float(dcg / ideal)


# EVOLVE-BLOCK-END
