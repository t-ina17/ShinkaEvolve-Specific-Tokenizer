```python
#!/usr/bin/env python3

"""jaccard-tokenizer の進化実行スクリプト。

このディレクトリの `initial.py`（Jaccard ベースの初期個体）を起点に、ShinkaEvolve の進化を起動します。

設計方針
- 実験条件（ESCI設定・NDCG@k・max_queries など）をこのファイル上部に集約して、再現性を上げる
- ESCI-data のパスは `--data_path` か環境変数 `ESCI_DATA_PATH` で外から指定できる
- 評価器は既存の `examples/specific-tokenizer/evaluate.py` を再利用する

前提
- LLM 実行に必要な環境変数（例: OPENAI_API_KEY）が有効である
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from shinka.core import EvolutionConfig, EvolutionRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Make paths robust to current working directory.
BASE_DIR = Path(__file__).resolve().parent
EVAL_PROGRAM_PATH = BASE_DIR / "evaluate.py"

# Amazon ESCI-data を使う場合は、repo root を data_path に指定します。
# 共有・論文掲載用に、絶対パスは CLI / 環境変数で上書きできるようにする。
# - 優先順位: --data_path > ESCI_DATA_PATH > DEFAULT_DATA_PATH
DEFAULT_DATA_PATH = "/Users/tina/work/study/sakana-ai/prod/data/esci-data"

# ESCI 評価条件（論文の再現条件として固定したい値をここに集約）
DEFAULT_ESCI_LOCALE = "jp"
DEFAULT_ESCI_SPLIT = "train"
DEFAULT_ESCI_VERSION = "large"
DEFAULT_MAX_QUERIES = 200
DEFAULT_MAX_PRODUCTS_PER_QUERY = 80
DEFAULT_SEED = 42
DEFAULT_PRODUCT_TEXT_FIELDS = "title_brand"
DEFAULT_NDCG_K = 10

SEARCH_TASK_SYS_MSG = """You are an expert in Japanese (and mixed JP/EN) tokenization for e-commerce search.

Goal
- Improve tokenizer + matching score so that relevant products rank higher for each query.
- The fitness is mean NDCG@10 over queries (if labels exist), otherwise mean match score.

Constraints
- Keep the code fast and robust; do not crash on weird inputs.
- Avoid heavyweight dependencies unless absolutely necessary.
- Prefer deterministic behavior.

High-impact ideas
- Normalize text (NFKC), unify variations (USB-C, Type-C, 全角/半角, hyphens)
- Handle units and numbers (500ml, 1m, 100W)
- Japanese tokenization: add subword/character n-grams for OOV robustness
- Downweight noisy tokens, upweight exact matches for short queries
"""

def _parse_args() -> argparse.Namespace:
    """CLI 引数をパースして返す。"""
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve evolution for jaccard-tokenizer")
    parser.add_argument(
        "--init_program_path",
        type=str,
        default=str(BASE_DIR / "initial.py"),
        help="Initial program path (default: initial.py).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(BASE_DIR / "results_jaccard_tokenizer_oss120b_large_q200"),
        help="Results directory (default: results_jaccard_tokenizer_oss120b_large_q200).",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="SQLite DB path (default: <results_dir>/evolution_db_*.sqlite)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "ESCI-data repo root. Overrides ESCI_DATA_PATH env var. "
            "(default: ESCI_DATA_PATH or a local absolute path)"
        ),
    )
    parser.add_argument(
        "--esci_split",
        type=str,
        default=DEFAULT_ESCI_SPLIT,
        help="ESCI split to optimize on: train|test (default: train).",
    )
    parser.add_argument("--num_generations", type=int, default=10)
    return parser.parse_args()

def _resolve_db_path(results_dir: Path, db_path: Optional[str]) -> Path:
    """DB パスを決める。

    指定がない場合は results_dir 配下に DB を置き、run を自己完結させる。
    """
    if db_path:
        return Path(db_path)
    return results_dir / "evolution_db_oss120b_large_q200.sqlite"

def _resolve_data_path(data_path: Optional[str]) -> str:
    """ESCI-data のルートディレクトリを決める。"""
    if data_path:
        return str(data_path)
    return os.environ.get("ESCI_DATA_PATH", DEFAULT_DATA_PATH)

def _build_job_config(*, data_path: str, esci_split: str) -> LocalJobConfig:
    """評価ジョブの設定を作る（evaluate.py + ESCI 設定）。"""
    return LocalJobConfig(
        eval_program_path=str(EVAL_PROGRAM_PATH),
        extra_cmd_args={
            "data_path": data_path,
            "dataset": "esci",
            "esci_locale": DEFAULT_ESCI_LOCALE,
            "esci_split": str(esci_split),
            "esci_version": DEFAULT_ESCI_VERSION,
            "max_queries": DEFAULT_MAX_QUERIES,
            "max_products_per_query": DEFAULT_MAX_PRODUCTS_PER_QUERY,
            "seed": DEFAULT_SEED,
            "product_text_fields": DEFAULT_PRODUCT_TEXT_FIELDS,
            "ndcg_k": DEFAULT_NDCG_K,
        },
    )

def _build_db_config(db_path: Path) -> DatabaseConfig:
    """進化 DB（SQLite）の設定を作る。"""
    return DatabaseConfig(
        db_path=str(db_path),
        num_islands=2,
        archive_size=60,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )

def _build_evo_config(
    *,
    init_program_path: str,
    results_dir: Path,
    num_generations: int,
) -> EvolutionConfig:
    """EvolutionConfig を組み立てる。"""
    return EvolutionConfig(
        task_sys_msg=SEARCH_TASK_SYS_MSG,
        patch_types=["diff", "full"],
        patch_type_probs=[0.7, 0.3],
        num_generations=int(num_generations),
        max_parallel_jobs=2,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=["openai/gpt-oss-120b"],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5],
            reasoning_efforts=["low", "medium"],
            max_tokens=8192,
        ),
        init_program_path=str(Path(init_program_path)),
        results_dir=str(results_dir),
    )

def main() -> None:
    """CLI エントリポイント。"""
    args = _parse_args()

    if not EVAL_PROGRAM_PATH.exists():
        raise SystemExit(f"evaluate.py が見つかりません: {EVAL_PROGRAM_PATH}")

    results_dir = Path(args.results_dir)
    db_path = _resolve_db_path(results_dir=results_dir, db_path=args.db_path)

    data_path = _resolve_data_path(args.data_path)
    if not Path(data_path).exists():
        raise SystemExit(
            "ESCI-data が見つかりません: "
            f"{data_path}. --data_path か ESCI_DATA_PATH を設定してください。"
        )

    runner = EvolutionRunner(
        evo_config=_build_evo_config(
            init_program_path=args.init_program_path,
            results_dir=results_dir,
            num_generations=int(args.num_generations),
        ),
        job_config=_build_job_config(data_path=data_path, esci_split=args.esci_split),
        db_config=_build_db_config(db_path=db_path),
        verbose=True,
    )
    runner.run()

if __name__ == "__main__":
    main()

```