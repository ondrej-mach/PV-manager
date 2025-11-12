"""Generate per-month bar plot comparing dumb control vs MPC costs.

This script reads the monthly summary CSV produced by ``src/benchmark.py``
(``output/benchmark_monthly.csv``) and creates a side-by-side bar chart
showing the net cost for the dumb baseline and the MPC controller in each
month. The resulting figure is written to
``output/benchmark_monthly_mpc_vs_dumb.png`` by default.

Usage examples:

    python src/plot_monthly_mpc_vs_dumb.py
    python src/plot_monthly_mpc_vs_dumb.py --input output/benchmark_monthly.csv \
        --output output/custom_plot.png

"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/benchmark_monthly.csv"),
        help="Path to the monthly summary CSV produced by benchmark.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark_monthly_mpc_vs_dumb.png"),
        help="Where to write the generated plot",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip rerunning benchmark.py before plotting",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force benchmark.py to refresh cached data before plotting",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
    default=Path("tools/benchmark.py"),
        help="Path to benchmark.py entry point",
    )
    return parser.parse_args()


def _validate_columns(frame: pd.DataFrame) -> None:
    required = {"month", "dumb_cost_eur", "opt_cost_eur"}
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Monthly summary missing required columns: " f"{missing_str}"
        )


def _ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _run_benchmark(script_path: Path, refresh: bool) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Benchmark script not found at {script_path}")

    env = os.environ.copy()
    env.pop("BENCHMARK_MONTH", None)
    env.pop("SEGMENT_DAYS", None)
    if refresh:
        env["BENCHMARK_REFRESH"] = "1"
    print(f"Running benchmark via {script_path}...")
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)


def main() -> int:
    args = _parse_args()

    if not args.skip_benchmark:
        try:
            _run_benchmark(args.benchmark_script, args.refresh_data)
        except subprocess.CalledProcessError as exc:
            print(f"Benchmark run failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode or 1
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to run benchmark: {exc}", file=sys.stderr)
            return 1

    if not args.input.exists():
        print(f"Monthly summary not found at {args.input}. Run benchmark.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    if df.empty:
        print("Monthly summary CSV is empty; nothing to plot.", file=sys.stderr)
        return 1

    _validate_columns(df)

    df_sorted = df.sort_values("month")
    months = df_sorted["month"].tolist()
    dumb_costs = df_sorted["dumb_cost_eur"].to_numpy()
    mpc_costs = df_sorted["opt_cost_eur"].to_numpy()

    positions = range(len(months))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([p - width / 2 for p in positions], dumb_costs, width=width, label="Dumb control", color="#1f77b4")
    ax.bar([p + width / 2 for p in positions], mpc_costs, width=width, label="MPC", color="#ff7f0e")

    ax.set_title("Monthly Net Cost: Dumb vs MPC")
    ax.set_ylabel("Net cost (EUR)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _ensure_output_dir(args.output)
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
