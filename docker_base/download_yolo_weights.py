#!/usr/bin/env python3
"""Dummy YOLO pose pass that forces Ultralytics to download model weights."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

# Import ultralytics before torch (project convention)
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def main(args):
    """Load the pose model, run one dummy frame, and cache the .pt for later use."""
    model_name = args.model_name
    print(f"Loading YOLO pose model {model_name} (downloads weights if missing)...")
    model = YOLO(model_name)

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    print("Running dummy pose pass...")
    model.track(dummy, persist=True, device="cpu", verbose=True, tracker="bytetrack.yaml")

    src = None
    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        src = ckpt_path
    elif os.path.isfile(model_name):
        src = model_name
    else:
        candidate = Path(SETTINGS["weights_dir"]) / Path(model_name).name
        if candidate.is_file():
            src = str(candidate)
    if src is None:
        raise FileNotFoundError(f"Could not locate downloaded weights for {model_name}")
    print(f"Weights available at {src}")

    # Cache under Ultralytics weights_dir so YOLO("checkpoints/<name>") finds it later
    weights_dir = Path(SETTINGS["weights_dir"])
    basename = Path(model_name).name
    cache_targets = [
        weights_dir / basename,
        weights_dir / "checkpoints" / basename,
    ]
    if args.output_path:
        cache_targets.append(Path(args.output_path))
    for dest in cache_targets:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
            print(f"Copied weights to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy YOLO pose pass to download Ultralytics weights.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolo26x-pose.pt",
        help="Ultralytics pose model name or path (downloaded if missing)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/yolo26x-pose.pt",
        help="Where to copy the downloaded weights",
    )
    args = parser.parse_args()
    main(args)
