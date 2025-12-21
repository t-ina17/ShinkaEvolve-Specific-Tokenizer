#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

# Headless environments (CI/servers) may not have a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from shinka.plots import plot_improvement, plot_lineage_tree
from shinka.utils.load_df import load_programs_to_df


def _resolve_db_path(results_dir: Path, db_path: str | None) -> Path:
    if db_path:
        return Path(db_path)

    candidates = sorted(results_dir.glob("evolution_db*.sqlite"))
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No evolution_db*.sqlite found under: {results_dir}. "
            "Pass --db explicitly."
        )

    raise FileExistsError(
        "Multiple DB candidates found. Pass --db explicitly: "
        + ", ".join(str(p) for p in candidates)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize Shinka evolution results (improvement curve + lineage tree)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_specific_tokenizer_oss120b_long",
        help="Results directory that contains evolution_db*.sqlite.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to the SQLite DB (overrides --results_dir auto-detect).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for images (default: <results_dir>/viz).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="specific-tokenizer",
        help="Plot title prefix.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    db_path = _resolve_db_path(results_dir, args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_programs_to_df(str(db_path))
    if df is None or df.empty:
        raise RuntimeError(
            "No programs loaded from DB. If the run is still in progress, "
            "try again after it finishes (or copy the DB file and visualize the copy)."
        )

    # Improvement curve
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    plot_improvement(df, title=f"{args.title}: Improvement", fig=fig1, ax=ax1)
    improvement_path = out_dir / "improvement.png"
    fig1.savefig(improvement_path, dpi=200)
    plt.close(fig1)

    # Lineage tree
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    plot_lineage_tree(df, title=f"{args.title}: Lineage Tree", fig=fig2, ax=ax2)
    lineage_path = out_dir / "lineage_tree.png"
    fig2.savefig(lineage_path, dpi=200)
    plt.close(fig2)

    print(f"Wrote: {improvement_path}")
    print(f"Wrote: {lineage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
