#!/usr/bin/env python3
"""
Image Restoration Pipeline - Execution Script (run.py)

Usage:
    python run.py <input-dir> <output-dir>

Requirements satisfied:
- Reads all .npy files from the input directory.
- Creates the output directory if it does not exist.
- Generates one restored .npy file for every input file with identical filename.
- Outputs grayscale arrays with shape (H, W) or (H, W, 1).
- Output values within [0, 1] with no NaN or Inf values.
- Restored images have target resolution (2x upscaled).
- Runs on NVIDIA GPU (CUDA) if available, falling back to CPU.
- Operates offline without external API keys or downloads.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.swinir_model import build_swinir


def load_restoration_model(weights_path: Path, device: torch.device):
    """
    Loads the trained SwinIR model and weights.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    variant = saved_cfg.get("swinir_variant", "small")
    in_channels = saved_cfg.get("in_channels", 1)
    out_channels = saved_cfg.get("out_channels", 1)
    window_size = saved_cfg.get("window_size", 8)
    upscale = saved_cfg.get("upscale", 2)

    model = build_swinir(
        variant=variant,
        in_channels=in_channels,
        out_channels=out_channels,
        window_size=window_size,
        upscale=upscale,
    )

    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, upscale


def process_npy_file(file_path: Path, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Reads a single .npy file, preprocesses it into tensor, runs model inference,
    and returns a clean, restored float32 grayscale numpy array with shape (H, W) in range [0, 1].
    """
    raw_data = np.load(file_path).astype(np.float32)

    # Handle input shape to produce (1, 1, H, W) tensor
    if raw_data.ndim == 2:
        # Shape: (H, W)
        inp_tensor = torch.from_numpy(raw_data).unsqueeze(0).unsqueeze(0)
    elif raw_data.ndim == 3:
        if raw_data.shape[2] == 1:
            # Shape: (H, W, 1) -> (1, 1, H, W)
            inp_tensor = torch.from_numpy(raw_data.transpose(2, 0, 1)).unsqueeze(0)
        elif raw_data.shape[0] == 1:
            # Shape: (1, H, W) -> (1, 1, H, W)
            inp_tensor = torch.from_numpy(raw_data).unsqueeze(0)
        elif raw_data.shape[2] == 3:
            # Shape: (H, W, 3) -> convert to 1-channel grayscale mean -> (1, 1, H, W)
            gray = raw_data.mean(axis=2)
            inp_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        elif raw_data.shape[0] == 3:
            # Shape: (3, H, W) -> convert to 1-channel grayscale mean -> (1, 1, H, W)
            gray = raw_data.mean(axis=0)
            inp_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported 3D array shape: {raw_data.shape} in {file_path}")
    else:
        raise ValueError(f"Unsupported array dimensions: {raw_data.ndim} in {file_path}")

    inp_tensor = inp_tensor.to(device, non_blocking=True)

    with torch.no_grad():
        out_tensor = model(inp_tensor)

    # Extract 2D restored grayscale image array
    restored = out_tensor.squeeze().cpu().numpy()

    # Ensure shape is (H, W)
    if restored.ndim == 3 and restored.shape[0] == 1:
        restored = restored.squeeze(0)

    # Clean NaNs, Infs, and clamp values strictly to [0, 1]
    restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
    restored = np.clip(restored, 0.0, 1.0).astype(np.float32)

    return restored


def main():
    parser = argparse.ArgumentParser(description="Image Restoration Pipeline Execution")
    parser.add_argument("input_dir", type=str, help="Directory containing input .npy files")
    parser.add_argument("output_dir", type=str, help="Directory to save output restored .npy files")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "models" / "model_best.pth"),
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    weights_path = Path(args.weights).resolve()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select execution device (NVIDIA GPU if available, else CPU)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("=" * 65)
    print("               SWINIR IMAGE RESTORATION INFERENCE")
    print("=" * 65)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Weights Path     : {weights_path}")
    print(f"Target Device    : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Load Model
    start_load = time.time()
    model, upscale = load_restoration_model(weights_path, device)
    print(f"Model loaded successfully in {time.time() - start_load:.2f}s (Upscale factor: {upscale}x)")

    # Find all .npy files in input directory
    input_files = sorted(list(input_dir.glob("*.npy")))
    if not input_files:
        print(f"Warning: No .npy files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(input_files)} .npy input file(s) to process.\n")

    t0 = time.time()
    for idx, file_path in enumerate(input_files, 1):
        restored_arr = process_npy_file(file_path, model, device)

        # Output path matching input filename
        out_path = output_dir / file_path.name
        np.save(out_path, restored_arr)

        if idx % 50 == 0 or idx == len(input_files):
            print(f"  [{idx}/{len(input_files)}] Saved: {out_path.name} | Shape: {restored_arr.shape} | Range: [{restored_arr.min():.4f}, {restored_arr.max():.4f}]")

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"Restoration Complete! Processed {len(input_files)} images in {total_time:.2f}s ({total_time / len(input_files):.4f}s/img)")
    print(f"Restored .npy outputs saved to: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
