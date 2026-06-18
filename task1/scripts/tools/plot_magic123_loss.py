#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def event_dirs(root: Path):
    if not root.exists():
        return []
    return sorted({event.parent for event in root.rglob("events.out.tfevents*")})


def read_scalars(root: Path, tag: str):
    values = []
    for event_dir in event_dirs(root):
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        values.extend((scalar.step, scalar.value) for scalar in accumulator.Scalars(tag))
    values.sort(key=lambda item: item[0])
    return values


def ema(values, alpha=0.15):
    smoothed = []
    current = None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def plot_metric(tag, filename, title, ylabel, coarse_run, fine_run, output_dir):
    series = [
        ("object_C_coarse", read_scalars(coarse_run, tag)),
        ("object_C_fine", read_scalars(fine_run, tag)),
    ]
    if not any(points for _, points in series):
        print(f"No TensorBoard scalar found for {tag}; skip {filename}.")
        return

    plt.figure(figsize=(8, 5), dpi=180)
    colors = {"object_C_coarse": "#d62728", "object_C_fine": "#1f77b4"}

    for name, points in series:
        if not points:
            continue
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        color = colors[name]
        plt.plot(steps, values, color=color, alpha=0.22, linewidth=1.0, label=f"{name} raw")
        plt.plot(steps, ema(values), color=color, alpha=0.95, linewidth=2.6, label=f"{name} EMA")

    plt.title(title, fontsize=13)
    plt.xlabel("Step", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote {output_path}")


def main():
    task1_root = Path(__file__).resolve().parents[2]
    output_dir = task1_root / "outputs" / "figures"
    coarse_run = task1_root / "submission_assets/object_C/magic123_output/object_C_coarse/run"
    fine_run = task1_root / "submission_assets/object_C/magic123_output/object_C_fine/run"
    metrics = {
        "train/loss": ("objectC_loss_total.png", "Object C Magic123 Loss Total", "magic123/loss_total"),
        "train/loss_rgb": ("objectC_loss_rgb.png", "Object C Magic123 Loss RGB", "magic123/loss_rgb"),
        "train/loss_mask": ("objectC_loss_mask.png", "Object C Magic123 Loss Mask", "magic123/loss_mask"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for tag, (filename, title, ylabel) in metrics.items():
        plot_metric(tag, filename, title, ylabel, coarse_run, fine_run, output_dir)


if __name__ == "__main__":
    main()
