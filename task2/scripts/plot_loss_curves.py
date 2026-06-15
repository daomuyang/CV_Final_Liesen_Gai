#!/usr/bin/env python3
"""Plot Action L1 loss curves from a WandB offline run or lerobot train log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _load_from_wandb_run(run_dir: Path) -> tuple[list[int], list[float]]:
    import wandb

    api = wandb.Api()
    run = api.from_path(str(run_dir))
    history = run.history(samples=10000, keys=["step", "l1_loss", "loss"])
    steps = history.get("step") or history.get("_step")
    l1 = history.get("l1_loss")
    if l1 is None:
        l1 = history.get("loss")
    if steps is None or l1 is None:
        raise ValueError(f"No l1_loss/loss in wandb run: {run_dir}")
    pairs = sorted(zip(steps, l1), key=lambda x: x[0])
    return [int(s) for s, _ in pairs], [float(v) for _, v in pairs]


def _load_from_log(log_path: Path) -> tuple[list[int], list[float]]:
    pattern = re.compile(r"step:(\d+)\s+.*loss:([\d.]+)")
    steps: list[int] = []
    losses: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise ValueError(f"No step/loss lines found in {log_path}")
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run", type=Path, default=None, help="Path to wandb offline run dir")
    parser.add_argument("--log-file", type=Path, default=None, help="Fallback: lerobot train stdout log")
    parser.add_argument("--label", type=str, default="ABC")
    parser.add_argument("--output", type=Path, default=Path("outputs/loss_curves_ABC.png"))
    args = parser.parse_args()

    if args.wandb_run is not None:
        steps, losses = _load_from_wandb_run(args.wandb_run)
    elif args.log_file is not None:
        steps, losses = _load_from_log(args.log_file)
    else:
        raise SystemExit("Provide --wandb-run or --log-file")

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, label=f"train {args.label}", linewidth=1.5)
    plt.xlabel("Training step")
    plt.ylabel("Action L1 loss")
    plt.title(f"ACT training loss — calvin_env_{args.label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output} ({len(steps)} points)")


if __name__ == "__main__":
    main()
