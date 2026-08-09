import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


def add_gaussian_noise(img: torch.Tensor, sigma_range: Tuple[float, float] = (0.0, 0.05)):
    sigma = random.uniform(*sigma_range)
    noise = torch.randn_like(img) * sigma
    return (img + noise).clamp(0.0, 1.0)


def apply_gaussian_blur(img: torch.Tensor, radius_range: Tuple[float, float] = (0.5, 2.0)):
    radius = random.uniform(*radius_range)
    pil = TF.to_pil_image(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return TF.to_tensor(pil)


def apply_random_downsampling(img: torch.Tensor, scale_range: Tuple[float, float] = (0.5, 0.9)):
    scale = random.uniform(*scale_range)
    _, H, W = img.shape
    new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))

    img_batch = img.unsqueeze(0)
    down = torch.nn.functional.interpolate(
        img_batch, size=(new_H, new_W), mode="bicubic", align_corners=False
    )
    up = torch.nn.functional.interpolate(
        down, size=(H, W), mode="bicubic", align_corners=False
    )
    return up.squeeze(0).clamp(0.0, 1.0)


def paired_random_crop(
    inp: torch.Tensor, tgt: torch.Tensor, crop_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, H, W = inp.shape
    if H <= crop_size or W <= crop_size:
        return inp, tgt
    scale_h = tgt.shape[1] // H
    scale_w = tgt.shape[2] // W
    top = random.randint(0, H - crop_size)
    left = random.randint(0, W - crop_size)
    inp = inp[:, top : top + crop_size, left : left + crop_size]
    top_tgt = top * scale_h
    left_tgt = left * scale_w
    crop_h_tgt = crop_size * scale_h
    crop_w_tgt = crop_size * scale_w
    tgt = tgt[:, top_tgt : top_tgt + crop_h_tgt, left_tgt : left_tgt + crop_w_tgt]
    return inp, tgt


def paired_random_flip_rotate(
    inp: torch.Tensor, tgt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() > 0.5:
        inp = TF.hflip(inp)
        tgt = TF.hflip(tgt)
    if random.random() > 0.5:
        inp = TF.vflip(inp)
        tgt = TF.vflip(tgt)
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        inp = torch.rot90(inp, k, dims=[1, 2])
        tgt = torch.rot90(tgt, k, dims=[1, 2])
    return inp, tgt


class PairedImageDataset(Dataset):

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        crop_size: Optional[int] = 256,
        augment: bool = True,
        noise_sigma: Optional[Tuple[float, float]] = (0.0, 0.05),
        blur_radius: Optional[Tuple[float, float]] = (0.3, 1.5),
        downsample: Optional[Tuple[float, float]] = (0.5, 0.9),
    ):
        assert split in self.VALID_SPLITS, f"split must be one of {self.VALID_SPLITS}"
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        self.noise_sigma = noise_sigma
        self.blur_radius = blur_radius
        self.downsample = downsample

        split_dir = Path(root_dir) / split
        self.input_dir = split_dir / "input"
        self.target_dir = split_dir / "target"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".npy"}
        self.input_paths = sorted(
            p for p in self.input_dir.iterdir() if p.suffix.lower() in extensions
        )

        self.has_target = self.target_dir.exists()
        if self.has_target:
            self.target_paths = sorted(
                p for p in self.target_dir.iterdir() if p.suffix.lower() in extensions
            )
        else:
            self.target_paths = []

        if split in ("train", "val"):
            if not self.has_target:
                raise FileNotFoundError(f"Target directory not found for split '{split}': {self.target_dir}")
            if len(self.input_paths) != len(self.target_paths):
                raise ValueError(
                    f"Mismatch for split '{split}': {len(self.input_paths)} inputs vs "
                    f"{len(self.target_paths)} targets."
                )
        elif self.has_target and len(self.target_paths) > 0:
            if len(self.input_paths) != len(self.target_paths):
                raise ValueError(
                    f"Mismatch for split '{split}': {len(self.input_paths)} inputs vs "
                    f"{len(self.target_paths)} targets."
                )

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.input_paths)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
            tensor = torch.from_numpy(data).float()
            tensor = torch.clamp(tensor, 0.0, 1.0)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            elif tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            return tensor
        else:
            img = Image.open(path).convert("RGB")
            return self.to_tensor(img)

    def __getitem__(self, idx: int) -> dict:
        inp = self._load_rgb(self.input_paths[idx])
        item = {
            "input": inp,
            "input_path": str(self.input_paths[idx]),
        }

        if len(self.target_paths) > 0:
            tgt = self._load_rgb(self.target_paths[idx])

            if self.augment and self.crop_size is not None:
                inp, tgt = paired_random_crop(inp, tgt, self.crop_size)

            if self.augment:
                inp, tgt = paired_random_flip_rotate(inp, tgt)

            item["target"] = tgt
            item["target_path"] = str(self.target_paths[idx])

        if self.augment:
            if self.blur_radius and random.random() > 0.5:
                inp = apply_gaussian_blur(inp, self.blur_radius)
            if self.downsample and random.random() > 0.5:
                inp = apply_random_downsampling(inp, self.downsample)
            if self.noise_sigma and random.random() > 0.5:
                inp = add_gaussian_noise(inp, self.noise_sigma)
            item["input"] = inp

        return item


def build_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 256,
    augment: bool = True,
    noise_sigma: Tuple[float, float] = (0.0, 0.05),
    blur_radius: Tuple[float, float] = (0.3, 1.5),
    downsample: Tuple[float, float] = (0.5, 0.9),
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    shared_kwargs = dict(
        noise_sigma=noise_sigma,
        blur_radius=blur_radius,
        downsample=downsample,
        crop_size=crop_size,
    )

    train_ds = PairedImageDataset(
        root_dir=data_root, split="train", augment=augment, **shared_kwargs
    )
    val_ds = PairedImageDataset(
        root_dir=data_root, split="val", augment=False, **shared_kwargs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def build_test_loader(
    data_root: str,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    test_ds = PairedImageDataset(
        root_dir=data_root,
        split="test",
        augment=False,
        crop_size=None,
    )
    return DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
