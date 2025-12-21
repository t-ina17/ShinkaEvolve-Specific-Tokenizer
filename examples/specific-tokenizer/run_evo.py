#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from shinka.core import EvolutionConfig, EvolutionRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Make paths robust to current working directory.
BASE_DIR = Path(__file__).resolve().parent

# Amazon ESCI-data を使う場合は、repo root を data_path に指定します。
# ユーザー環境: /Users/tina/work/study/sakana-ai/prod/data/esci-data
# 共有したい場合はこのパスを適宜変更してください。
TASK_DATA_PATH = "/Users/tina/work/study/sakana-ai/prod/data/esci-data"

job_config = LocalJobConfig(
    eval_program_path=str(BASE_DIR / "evaluate.py"),
    extra_cmd_args={
        "data_path": TASK_DATA_PATH,
        "dataset": "esci",
        # 日本語のみ
        "esci_locale": "jp",
        # まずは train で回して改善を探索（必要なら test で最終評価）
        "esci_split": "train",
        # task1相当の reduced set
        "esci_version": "large",
        # ある程度の規模で探索（必要に応じて調整）
        "max_queries": 200,
        "max_products_per_query": 80,
        "seed": 42,
        "product_text_fields": "title_brand",
        "ndcg_k": 10,
    },
)

# DBはローカルに作る
# - まず軽めの設定で動作確認し、必要に応じて増やす

db_config = DatabaseConfig(
    db_path=str(BASE_DIR / "evolution_db_oss120b_large_q200.sqlite"),
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

search_task_sys_msg = """You are an expert in Japanese (and mixed JP/EN) tokenization and domain-specific morphological analysis for e-commerce search.

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
- Split brand/model tokens (iPhone15, S24, JNL)
- Japanese tokenization: combine morpheme-like splits with character n-grams for OOV robustness
- Downweight noisy tokens, upweight exact matches for short queries
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.7, 0.3],
    # ある程度の世代数で探索
    num_generations=10,
    max_parallel_jobs=2,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    # ユーザー指定モデル
    llm_models=["openai/gpt-oss-120b"],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5],
        reasoning_efforts=["low", "medium"],
        max_tokens=8192,
    ),
    init_program_path=str(BASE_DIR / "initial.py"),
    results_dir=str(BASE_DIR / "results_specific_tokenizer_oss120b_large_q200"),
)

def main():
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
