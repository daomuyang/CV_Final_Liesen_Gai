#!/usr/bin/env python3
"""Plot eval report metrics for the homework PDF (chunk L1 curves, gap bars)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _effective_len(steps: list) -> int:
    last = -1
    for i, v in enumerate(steps):
        if v is not None:
            last = i
    return last + 1


def plot_chunk_curves(report: dict, output: Path, title: str) -> None:
    ind_steps = report["in_distribution"]["chunk_step_l1"]
    zs_steps = report["zero_shot"]["chunk_step_l1"]
    n = min(_effective_len(ind_steps), _effective_len(zs_steps))
    steps = list(range(n))

    ind_vals = [ind_steps[i] for i in range(n)]
    zs_vals = [zs_steps[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, ind_vals, "o-", label=f"In-dist ({report['in_distribution_eval']})", linewidth=2)
    ax.plot(steps, zs_vals, "s-", label=f"Zero-shot ({report['zero_shot_eval']})", linewidth=2)
    ax.set_xlabel("Action chunk step index")
    ax.set_ylabel("Mean per-frame L1 (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_gap_bars(reports: list[dict], labels: list[str], output: Path) -> None:
    ind_l1 = [r["in_distribution"]["action_l1_loss"] for r in reports]
    zs_l1 = [r["zero_shot"]["action_l1_loss"] for r in reports]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], ind_l1, width, label="In-distribution L1")
    ax.bar([i + width / 2 for i in x], zs_l1, width, label="Zero-shot L1 (D)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Action L1 (normalized)")
    ax.set_title("Visual Distribution Shift: In-dist vs Zero-shot")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval report metrics")
    parser.add_argument("--report", type=Path, action="append", help="report_*.json (repeatable)")
    parser.add_argument("--label", type=str, action="append", help="Label per report (same order)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.report:
        raise SystemExit("Provide at least one --report")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [_load(p) for p in args.report]
    labels = args.label or [r.get("train_split", f"run{i}") for i, r in enumerate(reports)]

    if len(labels) != len(reports):
        raise SystemExit("--label count must match --report count")

    for report, label in zip(reports, labels, strict=True):
        out = args.output_dir / f"chunk_l1_{label}.png"
        plot_chunk_curves(
            report,
            out,
            title=f"Action Chunking L1 by Horizon ({label})",
        )
        print(f"Wrote {out}")

    if len(reports) >= 2:
        out = args.output_dir / "gap_comparison.png"
        plot_gap_bars(reports, labels, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
