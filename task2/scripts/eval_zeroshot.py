#!/usr/bin/env python3
"""Offline action L1 evaluation for LeRobot ACT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.random_utils import set_seed

# CALVIN / LeRobot 7-dim action convention (for per-dim reporting).
ACTION_DIM_LABELS = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "gripper",
]


def _parse_episodes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return list(eval(raw))  # noqa: S307


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            out[key] = value
    return out


def _infer_split_name(eval_root: Path) -> str:
    name = eval_root.name
    if name.startswith("calvin_env_"):
        return name.removeprefix("calvin_env_")
    return name


def _success_rates(frame_l1: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    n = max(len(frame_l1), 1)
    return {f"success_rate_t{str(t).replace('.', '_')}": float((frame_l1 < t).sum() / n) for t in thresholds}


def _init_accumulators(chunk_size: int, action_dim: int) -> dict:
    return {
        "sum_l1": 0.0,
        "n_batches": 0,
        "n_frames": 0,
        "frame_l1_chunks": [],
        "chunk_step_sum": np.zeros(chunk_size, dtype=np.float64),
        "chunk_step_count": np.zeros(chunk_size, dtype=np.float64),
        "action_dim_sum": np.zeros(action_dim, dtype=np.float64),
        "action_dim_count": np.zeros(action_dim, dtype=np.float64),
    }


def _update_accumulators(
    acc: dict,
    l1_elems: torch.Tensor,
    pad_mask: torch.Tensor,
    action_dim: int,
) -> None:
    """Accumulate per-frame, per-chunk-step, and per-action-dim L1 stats."""
    valid_steps = pad_mask.squeeze(-1).sum(dim=-1).clamp_min(1)
    per_frame_l1 = (l1_elems.sum(dim=(-1, -2)) / (valid_steps * action_dim)).detach().cpu().numpy()
    acc["frame_l1_chunks"].append(per_frame_l1)

    denom = pad_mask.sum() * action_dim
    acc["sum_l1"] += float((l1_elems.sum() / denom.clamp_min(1)).item())
    acc["n_batches"] += 1
    acc["n_frames"] += int(per_frame_l1.shape[0])

    valid_2d = pad_mask.squeeze(-1).detach().cpu().numpy()  # (B, chunk)
    l1_np = l1_elems.detach().cpu().numpy()  # (B, chunk, dim)

    chunk_size = l1_np.shape[1]
    for step in range(chunk_size):
        step_valid = valid_2d[:, step]
        if step_valid.sum() == 0:
            continue
        step_l1 = l1_np[:, step, :].mean(axis=-1)  # mean over action dims
        acc["chunk_step_sum"][step] += float((step_l1 * step_valid).sum())
        acc["chunk_step_count"][step] += float(step_valid.sum())

    for dim in range(action_dim):
        dim_valid = valid_2d
        acc["action_dim_sum"][dim] += float((l1_np[:, :, dim] * dim_valid).sum())
        acc["action_dim_count"][dim] += float(dim_valid.sum())


def _finalize_metrics(
    acc: dict,
    thresholds: list[float],
    chunk_size: int,
    action_dim: int,
) -> dict:
    frame_l1 = (
        np.concatenate(acc["frame_l1_chunks"]) if acc["frame_l1_chunks"] else np.array([], dtype=np.float32)
    )
    success_metrics = _success_rates(frame_l1, thresholds)

    chunk_step_l1: list[float | None] = []
    for step in range(chunk_size):
        if acc["chunk_step_count"][step] > 0:
            chunk_step_l1.append(float(acc["chunk_step_sum"][step] / acc["chunk_step_count"][step]))
        else:
            chunk_step_l1.append(None)

    valid_chunk_steps = [v for v in chunk_step_l1 if v is not None]
    first_step_l1 = chunk_step_l1[0] if chunk_step_l1 and chunk_step_l1[0] is not None else None
    last_step_l1 = valid_chunk_steps[-1] if valid_chunk_steps else None
    horizon_degradation = (
        float(last_step_l1 / first_step_l1) if first_step_l1 and last_step_l1 and first_step_l1 > 1e-8 else None
    )

    action_dim_l1: dict[str, float] = {}
    for dim in range(action_dim):
        label = ACTION_DIM_LABELS[dim] if dim < len(ACTION_DIM_LABELS) else f"dim_{dim}"
        if acc["action_dim_count"][dim] > 0:
            action_dim_l1[label] = float(acc["action_dim_sum"][dim] / acc["action_dim_count"][dim])

    return {
        "action_l1_loss": acc["sum_l1"] / max(acc["n_batches"], 1),
        "action_l1_median": float(np.median(frame_l1)) if len(frame_l1) else None,
        "action_l1_mean": float(np.mean(frame_l1)) if len(frame_l1) else None,
        "action_l1_std": float(np.std(frame_l1)) if len(frame_l1) else None,
        "action_l1_p10": float(np.percentile(frame_l1, 10)) if len(frame_l1) else None,
        "action_l1_p90": float(np.percentile(frame_l1, 90)) if len(frame_l1) else None,
        "action_l1_p95": float(np.percentile(frame_l1, 95)) if len(frame_l1) else None,
        "chunk_step_l1": chunk_step_l1,
        "chunk_first_step_l1": first_step_l1,
        "chunk_last_step_l1": last_step_l1,
        "chunk_horizon_degradation": horizon_degradation,
        "action_dim_l1": action_dim_l1,
        **success_metrics,
    }


def run_eval(
    checkpoint: Path,
    eval_root: Path,
    train_split: str,
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    episodes: list[int] | None = None,
    max_batches: int | None = None,
    device: str = "cpu",
    video_backend: str = "pyav",
    tolerance_s: float = 0.02,
    success_thresholds: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """Run offline eval and return a metrics dict."""
    if success_thresholds is None:
        success_thresholds = [0.05, 0.1, 0.5, 1.0]

    set_seed(seed)
    torch_device = torch.device(device)

    ckpt_dir = checkpoint / "pretrained_model"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing pretrained_model under {checkpoint}")

    policy_cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(ckpt_dir)
    policy.to(torch_device)
    policy.eval()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=str(ckpt_dir)
    )

    eval_root = eval_root.resolve()
    eval_split = _infer_split_name(eval_root)
    repo_id = f"local/calvin_env_{eval_split}"

    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(eval_root))
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    dataset = LeRobotDataset(
        repo_id,
        root=str(eval_root),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    chunk_size = policy_cfg.chunk_size
    action_dim = dataset.meta.features[ACTION]["shape"][0]
    acc = _init_accumulators(chunk_size, action_dim)

    desc = f"Eval {train_split} -> {eval_split}"
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = _move_batch(batch, torch_device)
            batch = preprocessor(batch)

            actions_hat = policy.predict_action_chunk(batch)
            pad_mask = ~batch["action_is_pad"].unsqueeze(-1)
            l1_elems = torch.nn.functional.l1_loss(
                batch[ACTION], actions_hat, reduction="none"
            ) * pad_mask

            _update_accumulators(acc, l1_elems, pad_mask, action_dim)

    metrics = _finalize_metrics(acc, success_thresholds, chunk_size, action_dim)
    is_zero_shot = train_split.upper() != eval_split.upper()

    return {
        "train_split": train_split,
        "eval_split": eval_split,
        "eval_mode": "zero_shot" if is_zero_shot else "in_distribution",
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "eval_dataset": str(eval_root),
        "policy_chunk_size": policy_cfg.chunk_size,
        "policy_n_action_steps": policy_cfg.n_action_steps,
        "episodes": episodes,
        "num_batches": acc["n_batches"],
        "num_frames_evaluated": acc["n_frames"],
        "success_thresholds": success_thresholds,
        **metrics,
        "note": (
            "Offline metrics in preprocessor-normalized action space. "
            "chunk_step_l1: mean per-frame L1 at each step within the predicted action chunk "
            "(measures Action Chunking horizon degradation). "
            "chunk_horizon_degradation = last_step_l1 / first_step_l1 (>1 means error grows with horizon). "
            "For homework Success Rate (task-level), see docs/calvin_sim_setup.md."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ACT eval (in-distribution or zero-shot)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoints/XXXXXX")
    parser.add_argument("--eval-root", type=Path, default=Path("data/calvin_env_D"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--episodes", type=str, default=None, help='e.g. "[0,1,2]"')
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--tolerance-s", type=float, default=0.02)
    parser.add_argument(
        "--success-thresholds",
        type=str,
        default="0.05,0.1,0.5,1.0",
        help="Comma-separated normalized per-frame L1 thresholds",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="unknown", help="A / B / C / ABC (for logging)")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.success_thresholds.split(",") if x.strip()]
    results = run_eval(
        args.checkpoint,
        args.eval_root,
        args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        episodes=_parse_episodes(args.episodes),
        max_batches=args.max_batches,
        device=args.device,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
        success_thresholds=thresholds,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
