import math
from typing import Dict, List

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr_vals = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr_vals.mean()


def psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> List[float]:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    vals = 10.0 * torch.log10((max_val ** 2) / mse)
    return vals.tolist()


def _gaussian_kernel_metric(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.unsqueeze(0).unsqueeze(0)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    max_val: float = 1.0,
) -> torch.Tensor:
    C1 = (C1 * max_val) ** 2 if max_val != 1.0 else C1
    C2 = (C2 * max_val) ** 2 if max_val != 1.0 else C2

    B, C, H, W = pred.shape
    kernel = _gaussian_kernel_metric(kernel_size, sigma).to(pred.device)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    padding = kernel_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=C) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    return ssim_map.mean()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> List[float]:
    results = []
    for i in range(pred.shape[0]):
        val = ssim(pred[i : i + 1], target[i : i + 1], **kwargs)
        results.append(val.item())
    return results


class MetricEvaluator:

    def __init__(self):
        self.reset()

    def reset(self):
        self._psnr_values: List[float] = []
        self._ssim_values: List[float] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        self._psnr_values.extend(psnr_batch(pred, target))
        self._ssim_values.extend(ssim_batch(pred, target))

    def compute(self) -> Dict[str, float]:
        if not self._psnr_values:
            return {"psnr": 0.0, "ssim": 0.0}
        avg_psnr = sum(self._psnr_values) / len(self._psnr_values)
        avg_ssim = sum(self._ssim_values) / len(self._ssim_values)
        return {"psnr": avg_psnr, "ssim": avg_ssim}

    def __repr__(self) -> str:
        metrics = self.compute()
        return f"MetricEvaluator(PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f})"


if __name__ == "__main__":
    torch.manual_seed(0)
    pred   = torch.rand(4, 3, 256, 256)
    target = torch.rand(4, 3, 256, 256)

    p = psnr(pred, target)
    s = ssim(pred, target)
    print(f"PSNR: {p:.2f} dB  |  SSIM: {s:.4f}")

    evaluator = MetricEvaluator()
    evaluator.update(pred, target)
    evaluator.update(pred, target)
    print(evaluator)
    print("Metrics smoke-test passed")
