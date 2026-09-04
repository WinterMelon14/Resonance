"""Compare matching AMT evaluation JSON files from two result folders.

Put this file in ``backend/`` and run:

    python compare_amt_results.py

Edit the configuration constants below when comparing v1, v2, v3, etc.
No third-party packages are required.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


# CFG
BACKEND_DIR = Path(__file__).resolve().parent
BASELINE_DIR = BACKEND_DIR / "data" / "output" / "v1"
CANDIDATE_DIR = BACKEND_DIR / "data" / "output" / "v4"


REPORT_DIR = BACKEND_DIR / "data" / "output" / "comparisons"


FAIL_ON_MISSING_BASELINE = True

TIE_TOLERANCE = 1e-12

CONSOLE_METRICS = (
    "selected_thresholds.score",
    "metrics.frame_active.precision",
    "metrics.frame_active.recall",
    "metrics.frame_active.f1",
    "metrics.event_onset.f1",
    "metrics.event_offset.f1",
    "metrics.note.onset.f1",
    "metrics.note.onset_offset.f1",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    """Return dotted paths for all finite numeric leaves (excluding booleans)."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result[prefix] = number
    return result


def preference(metric: str) -> int:
    """Return +1 when higher is better, -1 when lower is better, 0 otherwise."""
    leaf = metric.rsplit(".", 1)[-1]
    if leaf in {"fp", "fn"}:
        return -1
    if leaf in {"tp", "precision", "recall", "f1", "score"}:
        return 1
    return 0  # frames and threshold values have no inherent winner


def verdict(metric: str, delta: float) -> str:
    direction = preference(metric)
    if direction == 0:
        return "changed" if abs(delta) > TIE_TOLERANCE else "tie"
    adjusted = direction * delta
    if adjusted > TIE_TOLERANCE:
        return "win"
    if adjusted < -TIE_TOLERANCE:
        return "loss"
    return "tie"


def display_number(value: float) -> str:
    if value.is_integer() and abs(value) < 1e12:
        return str(int(value))
    return f"{value:.6f}"


def main() -> int:
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(f"Baseline folder does not exist: {BASELINE_DIR}")
    if not CANDIDATE_DIR.is_dir():
        raise FileNotFoundError(f"Candidate folder does not exist: {CANDIDATE_DIR}")

    candidate_paths = sorted(CANDIDATE_DIR.glob("*.json"))
    if not candidate_paths:
        raise FileNotFoundError(f"No JSON files found in: {CANDIDATE_DIR}")

    missing: list[str] = []
    rows: list[dict[str, str | float]] = []
    paired_files = 0

    for candidate_path in candidate_paths:
        baseline_path = BASELINE_DIR / candidate_path.name
        if not baseline_path.is_file():
            missing.append(candidate_path.name)
            continue

        baseline = flatten_numbers(load_json(baseline_path))
        candidate = flatten_numbers(load_json(candidate_path))
        shared_metrics = sorted(baseline.keys() & candidate.keys())
        paired_files += 1

        for metric in shared_metrics:
            baseline_value = baseline[metric]
            candidate_value = candidate[metric]
            delta = candidate_value - baseline_value
            rows.append(
                {
                    "file": candidate_path.name,
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "delta": delta,
                    "verdict": verdict(metric, delta),
                }
            )

    if missing and FAIL_ON_MISSING_BASELINE:
        names = "\n  ".join(missing)
        raise FileNotFoundError(
            f"{len(missing)} candidate file(s) have no baseline match:\n  {names}"
        )
    if paired_files == 0:
        raise RuntimeError("No matching JSON filenames were found.")

    grouped: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metric"])].append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for metric in sorted(grouped):
        metric_rows = grouped[metric]
        baseline_values = [float(row["baseline"]) for row in metric_rows]
        candidate_values = [float(row["candidate"]) for row in metric_rows]
        verdicts = [str(row["verdict"]) for row in metric_rows]
        baseline_mean = fmean(baseline_values)
        candidate_mean = fmean(candidate_values)
        summary_rows.append(
            {
                "metric": metric,
                "files": len(metric_rows),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "mean_delta": candidate_mean - baseline_mean,
                "wins": verdicts.count("win"),
                "ties": verdicts.count("tie"),
                "losses": verdicts.count("loss"),
                "changed": verdicts.count("changed"),
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_name = BASELINE_DIR.name
    candidate_name = CANDIDATE_DIR.name
    stem = f"{candidate_name}_vs_{baseline_name}"
    detail_path = REPORT_DIR / f"{stem}_per_file.csv"
    summary_path = REPORT_DIR / f"{stem}_summary.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_by_metric = {str(row["metric"]): row for row in summary_rows}
    shown = [metric for metric in CONSOLE_METRICS if metric in summary_by_metric]

    print(f"\n{candidate_name} vs {baseline_name}: {paired_files} matched JSON files")
    if missing:
        print(f"Skipped {len(missing)} file(s) without a baseline match")
    print()
    print(
        f"{'metric':<38} {'baseline':>11} {'candidate':>11} "
        f"{'delta':>11} {'W/T/L':>11}"
    )
    print("-" * 87)
    for metric in shown:
        row = summary_by_metric[metric]
        delta = float(row["mean_delta"])
        print(
            f"{metric:<38} "
            f"{display_number(float(row['baseline_mean'])):>11} "
            f"{display_number(float(row['candidate_mean'])):>11} "
            f"{delta:>+11.6f} "
            f"{row['wins']}/{row['ties']}/{row['losses']:>3}"
        )

    print(f"\nPer-file report: {detail_path}")
    print(f"Summary report:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())