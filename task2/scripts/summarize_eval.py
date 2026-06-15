#!/usr/bin/env python3
"""Merge in-distribution and zero-shot eval JSONs into a report with gap metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gap(in_dist: float | None, zero_shot: float | None) -> dict:
    if in_dist is None or zero_shot is None:
        return {"absolute": None, "relative": None}
    absolute = zero_shot - in_dist
    relative = zero_shot / in_dist if in_dist > 1e-8 else None
    return {"absolute": absolute, "relative": relative}


def _effective_chunk_len(steps: list) -> int:
    last = -1
    for i, v in enumerate(steps):
        if v is not None:
            last = i
    return last + 1


def _chunk_step_gap(in_dist_steps: list, zero_shot_steps: list) -> list[dict]:
    n = min(
        _effective_chunk_len(in_dist_steps),
        _effective_chunk_len(zero_shot_steps),
        len(in_dist_steps),
        len(zero_shot_steps),
    )
    gaps = []
    for i in range(n):
        ind = in_dist_steps[i]
        zs = zero_shot_steps[i]
        if ind is None or zs is None:
            continue
        gaps.append(
            {
                "step": i,
                "in_dist": ind,
                "zero_shot": zs,
                "absolute_gap": zs - ind,
            }
        )
    return gaps


def build_report(in_dist: dict, zero_shot: dict) -> dict:
    train_split = in_dist.get("train_split") or zero_shot.get("train_split")
    eval_split_zero = zero_shot.get("eval_split", "D")

    l1_ind = in_dist.get("action_l1_loss")
    l1_zs = zero_shot.get("action_l1_loss")
    gap = _gap(l1_ind, l1_zs)

    chunk_gaps = _chunk_step_gap(
        in_dist.get("chunk_step_l1", []),
        zero_shot.get("chunk_step_l1", []),
    )

    # Visual distribution shift: how much worse is zero-shot vs in-dist at each horizon step.
    max_chunk_gap_step = None
    max_chunk_gap_val = None
    for item in chunk_gaps:
        g = item.get("absolute_gap")
        if g is not None and (max_chunk_gap_val is None or g > max_chunk_gap_val):
            max_chunk_gap_val = g
            max_chunk_gap_step = item["step"]

    return {
        "train_split": train_split,
        "in_distribution_eval": in_dist.get("eval_split"),
        "zero_shot_eval": eval_split_zero,
        "policy_chunk_size": in_dist.get("policy_chunk_size") or zero_shot.get("policy_chunk_size"),
        "policy_n_action_steps": in_dist.get("policy_n_action_steps")
        or zero_shot.get("policy_n_action_steps"),
        "generalization_gap": {
            "action_l1_loss": gap,
            "action_l1_median": _gap(in_dist.get("action_l1_median"), zero_shot.get("action_l1_median")),
            "chunk_horizon_degradation": _gap(
                in_dist.get("chunk_horizon_degradation"),
                zero_shot.get("chunk_horizon_degradation"),
            ),
        },
        "in_distribution": {
            "action_l1_loss": l1_ind,
            "action_l1_median": in_dist.get("action_l1_median"),
            "action_l1_p90": in_dist.get("action_l1_p90"),
            "chunk_first_step_l1": in_dist.get("chunk_first_step_l1"),
            "chunk_last_step_l1": in_dist.get("chunk_last_step_l1"),
            "chunk_horizon_degradation": in_dist.get("chunk_horizon_degradation"),
            "chunk_step_l1": in_dist.get("chunk_step_l1"),
            "action_dim_l1": in_dist.get("action_dim_l1"),
            "num_frames_evaluated": in_dist.get("num_frames_evaluated"),
        },
        "zero_shot": {
            "action_l1_loss": l1_zs,
            "action_l1_median": zero_shot.get("action_l1_median"),
            "action_l1_p90": zero_shot.get("action_l1_p90"),
            "chunk_first_step_l1": zero_shot.get("chunk_first_step_l1"),
            "chunk_last_step_l1": zero_shot.get("chunk_last_step_l1"),
            "chunk_horizon_degradation": zero_shot.get("chunk_horizon_degradation"),
            "chunk_step_l1": zero_shot.get("chunk_step_l1"),
            "action_dim_l1": zero_shot.get("action_dim_l1"),
            "num_frames_evaluated": zero_shot.get("num_frames_evaluated"),
        },
        "chunk_step_gap": chunk_gaps,
        "visual_shift_summary": {
            "max_chunk_step_gap": max_chunk_gap_val,
            "max_chunk_step_gap_at_step": max_chunk_gap_step,
            "interpretation": (
                "generalization_gap.action_l1_loss.absolute quantifies Visual Distribution Shift. "
                "chunk_step_gap shows at which action-chunk horizon step the shift hurts most. "
                "chunk_horizon_degradation > 1 means later chunk steps are harder (Action Chunking stress test)."
            ),
        },
        "source_files": {
            "in_distribution": str(in_dist.get("checkpoint", "")),
            "zero_shot": str(zero_shot.get("checkpoint", "")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize in-dist + zero-shot eval into report JSON")
    parser.add_argument("--in-dist", type=Path, required=True, help="eval_*_on_{TRAIN}.json")
    parser.add_argument("--zero-shot", type=Path, required=True, help="eval_*_on_D.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(_load(args.in_dist), _load(args.zero_shot))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
