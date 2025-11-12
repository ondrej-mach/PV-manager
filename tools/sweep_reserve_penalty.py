"""Sweep soft-reserve penalty values and plot resulting savings.

This script runs the benchmark for a grid of reserve soft-penalty weights
and records the reported savings percentage for each run. Results are saved
as CSV and plotted to output/reserve_penalty_sweep.png.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _run_benchmark(repo_root: Path, env: dict) -> pd.DataFrame:
    """Execute the benchmark via subprocess and return its summary row."""
    subprocess.run(
        [sys.executable, "tools/benchmark.py"],
        cwd=str(repo_root),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
    summary_path = repo_root / "output" / "benchmark_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("benchmark summary not found after run")
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError("benchmark summary is empty")
    return summary_df.iloc[[0]].copy()


def _build_penalty_grid(values: Iterable[str]) -> List[float | None]:
    grid: List[float | None] = []
    for raw in values:
        raw = raw.strip().lower()
        if raw in {"none", "default", "auto"}:
            grid.append(None)
            continue
        try:
            grid.append(float(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid penalty value '{raw}'") from exc
    if not grid:
        raise argparse.ArgumentTypeError("penalty grid must not be empty")
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-reserve",
        type=float,
        required=True,
        help="Soft reserve target as fraction of capacity (e.g. 0.4)",
    )
    parser.add_argument(
        "--benchmark-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month to restrict the benchmark window",
    )
    parser.add_argument(
        "--penalties",
        nargs="+",
        required=True,
        help="List of penalty weights in €/kWh (use 'none' to select default heuristic)",
    )
    parser.add_argument(
        "--hard-reserve",
        type=float,
        default=None,
        help="Optional hard SoC floor to apply (fraction of capacity)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output/reserve_penalty_sweep.csv",
        help="Where to store the aggregated sweep results",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force benchmark data refresh on first run",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    penalties = _build_penalty_grid(args.penalties)
    benchmark_month = args.benchmark_month or os.environ.get("BENCHMARK_MONTH")

    records = []
    for idx, penalty in enumerate(penalties):
        env = os.environ.copy()
        env["RESERVE_SOC_SOFT"] = str(args.soft_reserve)
        if args.hard_reserve is not None:
            env["RESERVE_SOC_HARD"] = str(args.hard_reserve)
        else:
            env.pop("RESERVE_SOC_HARD", None)
        if penalty is None:
            env.pop("RESERVE_SOFT_PENALTY_EUR_PER_KWH", None)
        else:
            env["RESERVE_SOFT_PENALTY_EUR_PER_KWH"] = str(penalty)
        if idx == 0 and args.refresh_data:
            env["BENCHMARK_REFRESH"] = "1"
        else:
            env.pop("BENCHMARK_REFRESH", None)
        if benchmark_month:
            env["BENCHMARK_MONTH"] = benchmark_month
        else:
            env.pop("BENCHMARK_MONTH", None)

        print(f"[SWEEP] Running benchmark for penalty={penalty if penalty is not None else 'default'}")
        summary_row = _run_benchmark(repo_root, env)
        summary_row["penalty_eur_per_kwh"] = penalty if penalty is not None else float("nan")
        summary_row["reserve_soc_soft"] = args.soft_reserve
        summary_row["reserve_soc_hard"] = args.hard_reserve if args.hard_reserve is not None else float("nan")
        records.append(summary_row)

    combined = pd.concat(records, ignore_index=True)
    combined_path = repo_root / args.output_csv
    combined.to_csv(combined_path, index=False)
    print(f"[SWEEP] Saved sweep results to {combined_path}")

    # Prepare data for plotting; drop NaN penalty (default heuristic) gracefully
    plot_df = combined.copy()
    plot_df = plot_df.dropna(subset=["penalty_eur_per_kwh"]).sort_values("penalty_eur_per_kwh")
    if plot_df.empty:
        print("[SWEEP] No numeric penalties provided; skipping plot generation")
        return

    penalties_plot = plot_df["penalty_eur_per_kwh"].to_numpy()
    savings_plot = plot_df["savings_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(penalties_plot, savings_plot, marker="o", linestyle="-")
    ax.set_xlabel("Soft reserve penalty (€/kWh)")
    ax.set_ylabel("Savings (%)")
    ax.set_title(f"Savings vs. soft reserve penalty (soft reserve={args.soft_reserve:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "reserve_penalty_sweep.png"
    fig.savefig(plot_path, dpi=150)
    print(f"[SWEEP] Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
