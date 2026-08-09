# SwinIR Image Restoration Pipeline

This repository contains a PyTorch implementation of the **SwinIR** (Swin Transformer for Image Restoration) pipeline, designed for image-to-image regression tasks such as denoising and super-resolution. It also includes a UNet baseline.

## Project Structure

```
I4E-Hackathon-2026-PS-Soln/
├── config/                  # YAML configuration files for train and eval
├── data/                    # Dataset directory (.npy array files)
│   ├── train/               # Train split (input/ and target/)
│   ├── val/                 # Validation split (input/ and target/)
│   └── test/                # Test split (input/ and target/)
├── datasets/                # Custom PyTorch Dataset and DataLoader logic
├── losses/                  # Combined loss function (L1, SSIM, Perceptual)
├── metrics/                 # PSNR and SSIM evaluators
├── models/                  # PyTorch model definitions (SwinIR & UNet)
├── outputs/                 # Checkpoints, TensorBoard logs, and saved results
└── scripts/                 # Training and Evaluation entrypoints
```

## Setup & Dependencies

The pipeline requires PyTorch, TorchVision, and Einops. Ensure your virtual environment is active, then install the required dependencies:

```bash
pip install -r requirements.txt
```
*(If you are using the local `semicon` venv, it is already set up with all dependencies.)*

## Dataset Format

The dataloader has been explicitly updated to support `(128, 128)` `float32` `.npy` array files. The single-channel `.npy` data is automatically tiled to 3 channels to be seamlessly consumed by the SwinIR architecture. 

The data should be organized as follows:
- `data/train/input/` and `data/train/target/`
- `data/val/input/` and `data/val/target/`
- `data/test/input/` and `data/test/target/`

## How to Run

### 1. Training

To train the SwinIR model, use the `train.py` script. It will automatically read the hyperparameters from `config/train_config.yaml`.

Run this from the project root directory:

```bash
.\semicon\Scripts\python scripts\train.py --config config\train_config.yaml
```

**Training Outputs:**
- Model weights will be saved to `outputs/checkpoints/model_best.pth` and `model_latest.pth`.
- Training logs (losses, metrics) will be saved in `outputs/logs/`.

*(Note: In `config/train_config.yaml`, the `epochs` variable currently defines how long training will run. You can adjust this for quicker tests).*

### 2. Evaluation

To evaluate a trained model on the test set, run the `evaluate.py` script:

```bash
.\semicon\Scripts\python scripts\evaluate.py --config config\eval_config.yaml
```

**Evaluation Outputs:**
- Metrics (PSNR, SSIM) will be printed to the terminal.
- Reconstructed outputs (e.g. enhanced `.npy` output converted to images) will be saved in `outputs/test_outputs/`.
