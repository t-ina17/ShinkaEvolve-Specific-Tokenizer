"""specific-tokenizer 用の評価スクリプト。

このディレクトリの `initial.py`（Jaccard ベースの初期個体）を評価するためのスクリプトです。

対応する入力データ
- CSV（手元サンプル/独自データ）
- Amazon ESCI Shopping Queries Dataset（ESCI-data）のリポジトリルート
    - `shopping_queries_dataset/*.parquet` を想定

評価の流れ
- 対象プログラム（initial.py / best/main.py など）から `run_experiment(examples, ndcg_k)` を呼ぶ
- 実行は `run_shinka_eval` でサンドボックス化し、結果を `metrics.json` に集約する

出力
- `<results_dir>/metrics.json` : 公開用メトリクス
- `<results_dir>/extra.json`   : 追加診断（token など、プログラムが返した dict 全体）

注意
- 実装は `examples/specific-tokenizer/evaluate.py` をベースにしており、互換のI/Fを維持しています。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval


Example = Dict[str, Any]
RunOutput = Dict[str, Any]
Metrics = Dict[str, Any]


def _load_csv_examples(
    data_path: str,
    max_rows: Optional[int] = None,
) -> List[Example]:
    """CSVから評価用の行データを読み込む。

    CSV列の例
    - query / product（または product_title）
    - label（任意: 数値。無い場合は教師なし評価になる）
    - query_id（任意。無い場合は query 文字列を代用）
    """
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row:
                continue
            query = (row.get("query") or "").strip()
            product = (row.get("product") or row.get("product_title") or "").strip()
            if not query or not product:
                continue

            label_raw = row.get("label", None)
            label: Optional[float]
            if label_raw is None or str(label_raw).strip() == "":
                label = None
            else:
                try:
                    label = float(label_raw)
                except Exception:
                    label = None

            query_id = row.get("query_id") or row.get("qid") or query

            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "product": product,
                    "label": label,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError(f"No usable rows loaded from {data_path}")
    return rows


def _pick_product_text(row: dict, product_text_fields: str) -> str:
    """ESCIの product テキストを、設定に応じて1本の文字列にまとめる。"""
    title = (row.get("product_title") or "").strip()
    brand = (row.get("product_brand") or "").strip()
    color = (row.get("product_color") or "").strip()
    desc = (row.get("product_description") or "").strip()
    bullet = (row.get("product_bullet_point") or "").strip()

    if product_text_fields == "title":
        return title
    if product_text_fields == "title_brand":
        parts = [title, brand, color]
        return " ".join([p for p in parts if p])
    if product_text_fields == "all":
        parts = [title, brand, color, bullet, desc]
        return " \n".join([p for p in parts if p])
    raise ValueError(
        f"Unknown product_text_fields={product_text_fields!r} (expected: title|title_brand|all)"
    )


def _esci_label_to_relevance(esci_label: Optional[str]) -> Optional[float]:
    """ESCIラベル（E/S/C/I）を NDCG 用の関連度スコアへ変換する。"""
    if esci_label is None:
        return None
    label = str(esci_label).strip().upper()
    if label == "E":
        return 3.0
    if label == "S":
        return 2.0
    if label == "C":
        return 1.0
    if label == "I":
        return 0.0
    return None


def _load_esci_examples(
    esci_root: str,
    locale: str = "jp",
    split: str = "train",
    version: str = "small",
    max_rows: Optional[int] = None,
    max_queries: Optional[int] = None,
    max_products_per_query: Optional[int] = None,
    seed: int = 42,
    product_text_fields: str = "title",
) -> List[Example]:
    """ESCI-data から評価用の example を作る。

    `esci_root` は `shopping_queries_dataset/` を含むディレクトリ（リポジトリルート）を指す。
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "pandas is required to load ESCI-data. Install it in your environment."
        ) from e

    root = Path(esci_root)
    sqd_dir = root / "shopping_queries_dataset"
    if not sqd_dir.exists():
        raise ValueError(
            f"shopping_queries_dataset/ not found under {esci_root}. "
            "Set --data_path to the ESCI-data repo root."
        )

    examples_path = sqd_dir / "shopping_queries_dataset_examples.parquet"
    products_path = sqd_dir / "shopping_queries_dataset_products.parquet"
    if not examples_path.exists() or not products_path.exists():
        raise ValueError(
            "Missing required parquet files under shopping_queries_dataset/. "
            f"Expected: {examples_path} and {products_path}"
        )

    try:
        df_examples = pd.read_parquet(examples_path)
        df_products = pd.read_parquet(products_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read parquet files. You likely need a parquet engine (pyarrow). "
            "Try: `uv pip install pyarrow` (or `pip install pyarrow`)."
        ) from e

    # Merge to attach product text.
    df = df_examples.merge(
        df_products,
        how="left",
        on=["product_locale", "product_id"],
        suffixes=("", ""),
    )

    locale = str(locale).strip().lower()
    split = str(split).strip().lower()
    version = str(version).strip().lower()

    # Filter locale
    df = df[df["product_locale"].astype(str).str.lower() == locale]

    # Filter version
    if version == "small":
        df = df[df["small_version"] == 1]
    elif version == "large":
        df = df[df["large_version"] == 1]
    else:
        raise ValueError("--version must be 'small' or 'large'")

    # Filter split
    if split in {"train", "test"}:
        df = df[df["split"].astype(str).str.lower() == split]
    else:
        raise ValueError("--split must be 'train' or 'test'")

    # Basic cleanup
    df = df.dropna(
        subset=["query", "product_title", "query_id", "product_id"], how="any"
    )

    # Optionally sample queries for speed
    rng = random.Random(seed)
    all_qids = list(df["query_id"].astype(str).unique())
    if max_queries is not None and max_queries > 0 and len(all_qids) > max_queries:
        picked = set(rng.sample(all_qids, k=max_queries))
        df = df[df["query_id"].astype(str).isin(picked)]

    # Optionally cap products per query for speed
    if max_products_per_query is not None and max_products_per_query > 0:
        # Deterministic shuffle within each query
        df = df.copy()
        df["__rand"] = [rng.random() for _ in range(len(df))]
        df = (
            df.sort_values(["query_id", "__rand"])
            .groupby("query_id")
            .head(max_products_per_query)
        )

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    records: List[Example] = []
    for _, row in df.iterrows():
        row_d = row.to_dict()
        query = str(row_d.get("query") or "").strip()
        qid = str(row_d.get("query_id") or "").strip()
        product_text = _pick_product_text(
            row_d, product_text_fields=product_text_fields
        ).strip()
        if not query or not qid or not product_text:
            continue

        label = _esci_label_to_relevance(row_d.get("esci_label"))
        records.append(
            {
                "query_id": qid,
                "query": query,
                "product": product_text,
                "label": label,
            }
        )

    if not records:
        raise ValueError(
            "No usable JP rows were produced. Check locale/split/version and parquet contents."
        )
    return records


def _load_examples_auto(
    data_path: str,
    dataset: str,
    max_rows: Optional[int],
    esci_locale: str,
    esci_split: str,
    esci_version: str,
    max_queries: Optional[int],
    max_products_per_query: Optional[int],
    seed: int,
    product_text_fields: str,
) -> List[Example]:
    """`--dataset` と `data_path` から入力形式を決めて読み込む。"""
    p = Path(data_path)
    ds = dataset.strip().lower()
    if ds == "auto":
        if p.is_dir():
            ds = "esci"
        else:
            ds = "csv"

    if ds == "csv":
        return _load_csv_examples(data_path=str(p), max_rows=max_rows)
    if ds == "esci":
        return _load_esci_examples(
            esci_root=str(p),
            locale=esci_locale,
            split=esci_split,
            version=esci_version,
            max_rows=max_rows,
            max_queries=max_queries,
            max_products_per_query=max_products_per_query,
            seed=seed,
            product_text_fields=product_text_fields,
        )
    raise ValueError("--dataset must be auto|csv|esci")


def validate_output(run_output: RunOutput) -> Tuple[bool, Optional[str]]:
    """対象プログラムが返す dict（run_experimentの戻り値）を検証する。"""
    if not isinstance(run_output, dict):
        return False, "run_experiment must return a dict"

    score = run_output.get("combined_score", None)
    if score is None:
        return False, "Missing combined_score"
    try:
        s = float(score)
    except Exception:
        return False, f"combined_score is not a float: {score!r}"

    if not math.isfinite(s):
        return False, f"combined_score is not finite: {s}"
    return True, None


def aggregate(results: List[RunOutput], results_dir: str) -> Metrics:
    """`run_shinka_eval` の生結果を、論文/可視化で扱いやすい形に集約する。"""
    if not results:
        return {"combined_score": 0.0, "public": {}, "private": {"error": "no_results"}}

    out = results[0]
    score = float(out.get("combined_score", 0.0))

    public = {
        "mode": out.get("mode"),
        "combined_score": score,
        "num_rows": out.get("num_rows"),
        "num_queries": out.get("num_queries"),
        "ndcg_k": out.get("ndcg_k"),
    }

    preview = out.get("preview")
    if isinstance(preview, list):
        public["preview"] = preview[:5]

    metrics: Metrics = {
        "combined_score": score,
        "public": public,
        "private": {},
    }

    # 追加情報を保存
    extra_path = os.path.join(results_dir, "extra.json")
    try:
        with open(extra_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        metrics["private"]["extra_save_error"] = str(e)

    return metrics


def _parse_args() -> argparse.Namespace:
    """CLI 引数を定義してパースする。"""
    parser = argparse.ArgumentParser(description="Evaluate tokenizer match fitness")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="auto|csv|esci. If data_path is a directory, auto assumes esci.",
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--max_products_per_query", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ndcg_k", type=int, default=10)

    # ESCI options (used when --dataset esci)
    parser.add_argument("--esci_locale", type=str, default="jp")
    parser.add_argument("--esci_split", type=str, default="train")
    parser.add_argument("--esci_version", type=str, default="small")
    parser.add_argument(
        "--product_text_fields",
        type=str,
        default="title",
        help="title|title_brand|all",
    )
    return parser.parse_args()


def main() -> None:
    """CLI エントリポイント。"""
    args = _parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    examples = _load_examples_auto(
        data_path=args.data_path,
        dataset=args.dataset,
        max_rows=args.max_rows,
        esci_locale=args.esci_locale,
        esci_split=args.esci_split,
        esci_version=args.esci_version,
        max_queries=args.max_queries,
        max_products_per_query=args.max_products_per_query,
        seed=args.seed,
        product_text_fields=args.product_text_fields,
    )

    def get_kwargs(_run_index: int) -> Dict[str, Any]:
        return {"examples": examples, "ndcg_k": int(args.ndcg_k)}

    def agg_with_context(r: List[dict]) -> Dict[str, Any]:
        return aggregate(r, results_dir=args.results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=args.program_path,
        results_dir=args.results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate_output,
        aggregate_metrics_fn=agg_with_context,
    )

    print("Correct:", correct)
    if not correct:
        print("Error:", error_msg)
    print("Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
