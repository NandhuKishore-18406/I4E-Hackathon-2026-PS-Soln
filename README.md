# Image Restoration Pipeline (SwinIR)

This repository contains an offline-ready PyTorch implementation of the **SwinIR** (Swin Transformer for Image Restoration) pipeline for image super-resolution, denoising, and deblurring.

## Directory Structure

```
<team_name>/
├── run.py                 # Main execution entrypoint for inference
├── requirements.txt       # Dependencies with version specifications
├── README.md              # Setup and execution instructions
└── models/                # PyTorch model definitions and trained weights
    ├── model_best.pth     # Trained SwinIR model checkpoint
    ├── swinir_model.py    # SwinIR network architecture
    └── __init__.py
```

## Setup & Requirements

### Dependencies

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 2.1.0`
- `torchvision >= 0.16.0`
- `numpy >= 1.24.0`
- `einops >= 0.7.0`
- `Pillow >= 10.0.0`
- `PyYAML >= 6.0`

## Execution Instructions

Run the solution using the command:

```bash
python run.py <input-dir> <output-dir>
```

### Example Usage:

```bash
python run.py ./data/test/input ./outputs/restored_output
```

### Pipeline Key Features & Verification:

1. **Input Handling**: Reads all `.npy` array files from `<input-dir>`. Supports 2D `(H, W)` and 3D `(H, W, 1)` array shapes.
2. **Directory Creation**: Automatically creates `<output-dir>` if it does not exist.
3. **Filename Preservation**: Generates one restored `.npy` file for every input file using the exact same filename.
4. **Grayscale Output Format**: Output arrays are single-channel grayscale arrays with target resolution `(H_out, W_out)` (2x upscaled from input resolution).
5. **Value Sanitization**: All output values are strictly clipped within `[0.0, 1.0]` float32 with zero `NaN` or `Inf` values (`np.nan_to_num`).
6. **Self-Contained & Offline Execution**: Automatically detects and leverages NVIDIA GPU (CUDA) if available. Operates completely offline without requiring internet access, external API keys, model downloads, or user interaction.
