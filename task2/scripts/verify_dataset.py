#!/usr/bin/env python3
"""Quick sanity check for the five calvin_env_* datasets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SPLITS = ("A", "B", "C", "ABC", "D")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    ok = True

    print(f"Data root: {data_root}")
    for split in SPLITS:
        ds_dir = data_root / f"calvin_env_{split}"
        info_path = ds_dir / "meta" / "info.json"
        if not info_path.exists():
            print(f"[FAIL] calvin_env_{split}: missing {info_path}")
            ok = False
            continue

        with open(info_path) as f:
            info = json.load(f)

        try:
            ds = LeRobotDataset(
                f"local/calvin_env_{split}",
                root=str(ds_dir),
                episodes=[0],
                tolerance_s=0.02,
                video_backend="pyav",
            )
            sample = ds[0]
            n_frames = len(ds)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] calvin_env_{split}: {exc}")
            ok = False
            continue

        print(
            f"[OK] calvin_env_{split}: "
            f"episodes={info['total_episodes']} frames(subset)={n_frames} "
            f"action={tuple(sample['action'].shape)} "
            f"image={tuple(sample['observation.images.image'].shape)}"
        )

    if ok:
        print("\nAll datasets passed verification.")
        return 0
    print("\nSome datasets failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
