#!/usr/bin/env python3
"""
Dataset Setup Helper Script (scripts/setup_dataset.py)

Populates data/train and data/val directories from existing dataset archives
(e.g., train(1)/train/NoisyLR and GT) if data/ is empty.
"""

import os
import sys
import glob
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def setup_dataset():
    data_dir = PROJECT_ROOT / "data"
    train_input = data_dir / "train" / "input"
    train_target = data_dir / "train" / "target"
    val_input = data_dir / "val" / "input"
    val_target = data_dir / "val" / "target"

    for d in [train_input, train_target, val_input, val_target]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if train input already has .npy files
    existing_train = list(train_input.glob("*.npy"))
    if len(existing_train) > 0:
        print(f"[Dataset Setup] Dataset already populated ({len(existing_train)} train samples found).")
        return

    # Search for candidate dataset source folders
    candidate_sources = [
        PROJECT_ROOT / "train(1)" / "train",
        PROJECT_ROOT / "train(1)",
        PROJECT_ROOT / "train",
        PROJECT_ROOT / "dataseti4e" / "train",
    ]

    source_dir = None
    for src in candidate_sources:
        if (src / "NoisyLR").exists() and (src / "GT").exists():
            source_dir = src
            break

    if source_dir is None:
        print("[Dataset Setup] Warning: Could not find NoisyLR and GT source dataset folders.")
        return

    noisy_files = sorted(list((source_dir / "NoisyLR").glob("*.npy")))
    gt_files = sorted(list((source_dir / "GT").glob("*.npy")))

    if not noisy_files or len(noisy_files) != len(gt_files):
        print(f"[Dataset Setup] Warning: Inconsistent dataset files in {source_dir}")
        return

    print(f"[Dataset Setup] Found {len(noisy_files)} sample pairs in {source_dir}")

    # Split: 90% Train, 10% Validation
    val_count = max(1, int(len(noisy_files) * 0.1))
    train_count = len(noisy_files) - val_count

    train_noisy = noisy_files[:train_count]
    train_gt = gt_files[:train_count]
    val_noisy = noisy_files[train_count:]
    val_gt = gt_files[train_count:]

    print(f"[Dataset Setup] Populating data/: {len(train_noisy)} Train samples | {len(val_noisy)} Val samples...")

    def link_or_copy_files(src_list, dst_dir):
        for src_path in src_list:
            dst_path = dst_dir / src_path.name
            if not dst_path.exists():
                try:
                    os.symlink(src_path, dst_path)
                except Exception:
                    shutil.copy2(src_path, dst_path)

    link_or_copy_files(train_noisy, train_input)
    link_or_copy_files(train_gt, train_target)
    link_or_copy_files(val_noisy, val_input)
    link_or_copy_files(val_gt, val_target)

    print(f"[Dataset Setup] Successfully populated dataset in '{data_dir}'!")

if __name__ == "__main__":
    setup_dataset()
