import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from torchvision import models


class L1Loss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


class SSIMLoss(nn.Module):

    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        kernel = _gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_size // 2

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self.kernel.expand(C, 1, -1, -1)

        mu_x = F.conv2d(x, kernel, padding=self.padding, groups=C)
        mu_y = F.conv2d(y, kernel, padding=self.padding, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, kernel, padding=self.padding, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, kernel, padding=self.padding, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, kernel, padding=self.padding, groups=C) - mu_xy

        num = (2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return num / den

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):

    DEFAULT_LAYERS = {9: 0.5, 18: 0.5}

    def __init__(
        self,
        layer_weights: dict = None,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.layer_weights = layer_weights or self.DEFAULT_LAYERS
        self.normalize_input = normalize_input

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_features = vgg.features

        max_layer = max(self.layer_weights.keys())
        self.vgg_submodel = nn.Sequential(*list(vgg_features.children())[: max_layer + 1])

        for param in self.vgg_submodel.parameters():
            param.requires_grad = False

        self._layer_outputs: dict = {}
        self._hooks = []
        for layer_idx in self.layer_weights:
            hook = list(self.vgg_submodel.children())[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        if normalize_input:
            self.register_buffer(
                "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._layer_outputs[layer_idx] = output
        return hook

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            pred = self._normalize(pred)
            target = self._normalize(target)

        self._layer_outputs.clear()
        _ = self.vgg_submodel(pred)
        pred_feats = {k: v for k, v in self._layer_outputs.items()}

        self._layer_outputs.clear()
        with torch.no_grad():
            _ = self.vgg_submodel(target)
        target_feats = {k: v for k, v in self._layer_outputs.items()}

        loss = torch.tensor(0.0, device=pred.device)
        for layer_idx, weight in self.layer_weights.items():
            loss = loss + weight * F.l1_loss(pred_feats[layer_idx], target_feats[layer_idx])
        return loss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        w_l1: float = 0.5,
        w_ssim: float = 0.3,
        w_perceptual: float = 0.2,
    ):
        super().__init__()
        assert abs(w_l1 + w_ssim + w_perceptual - 1.0) < 1e-4, \
            "Loss weights should sum to 1.0"

        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual

        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss() if w_perceptual > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim(pred, target)
        if self.w_perceptual > 0 and self.perceptual is not None:
            perceptual_val = self.perceptual(pred, target)
        else:
            perceptual_val = torch.tensor(0.0, device=pred.device)

        total = (
            self.w_l1 * l1_val
            + self.w_ssim * ssim_val
            + self.w_perceptual * perceptual_val
        )

        loss_dict = {
            "loss_total": total.item(),
            "loss_l1": l1_val.item(),
            "loss_ssim": ssim_val.item(),
            "loss_perceptual": perceptual_val.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    pred   = torch.rand(2, 3, 256, 256, requires_grad=True)
    target = torch.rand(2, 3, 256, 256)

    loss_fn = CombinedLoss()
    total, breakdown = loss_fn(pred, target)
    total.backward()

    print("Loss breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.4f}")
    print(f"  grad_norm: {pred.grad.norm().item():.6f}")
    print("Loss smoke-test passed")
