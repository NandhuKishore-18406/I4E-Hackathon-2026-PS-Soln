import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_loader import build_dataloaders
from losses.loss import CombinedLoss
from metrics.eval_metrics import MetricEvaluator


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def build_model(model_name: str, cfg: dict) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "swinir":
        from models.swinir_model import build_swinir
        model = build_swinir(
            variant=cfg.get("swinir_variant", "small"),
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            window_size=cfg.get("window_size", 8),
            upscale=cfg.get("upscale", 1),
            drop_rate=cfg.get("drop_rate", 0.0),
            attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        )
        return model

    elif model_name == "unet":
        from models.unet_model import UNet
        model = UNet(
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
        )
        return model

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'swinir' or 'unet'.")


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return start_epoch, best_psnr


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: AdamW,
    loss_fn: CombinedLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    logger: logging.Logger,
) -> dict:
    model.train()
    total_loss = 0.0
    log_every = cfg.get("log_every_n_steps", 50)
    clip_grad = cfg.get("grad_clip", None)

    for step, batch in enumerate(loader):
        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=cfg.get("amp", True)):
            pred = model(inp)
            loss, loss_dict = loss_fn(pred, tgt)

        scaler.scale(loss).backward()

        if clip_grad is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] Step [{step + 1}/{len(loader)}] "
                f"lr={lr:.6f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            )

    return {"train_loss": total_loss / len(loader)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: CombinedLoss,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    evaluator = MetricEvaluator()
    total_loss = 0.0

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        with autocast(enabled=cfg.get("amp", True)):
            pred = model(inp)
            loss, _ = loss_fn(pred, tgt)

        total_loss += loss.item()
        evaluator.update(pred.float(), tgt.float())

    metrics = evaluator.compute()
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Image Restoration Training")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"] = args.model

    model_name = cfg.get("model", "swinir")

    run_name = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = PROJECT_ROOT / cfg.get("output_dir", "outputs") / "checkpoints" / run_name
    log_dir = PROJECT_ROOT / cfg.get("log_dir", "outputs/logs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"Run: {run_name}  |  Model: {model_name}")
    logger.info(f"Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = cfg.get("data_root", "data")
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("num_workers", 4),
        crop_size=cfg.get("crop_size", 256),
        augment=cfg.get("augment", True),
        noise_sigma=tuple(cfg.get("noise_sigma", [0.0, 0.05])),
        blur_radius=tuple(cfg.get("blur_radius", [0.3, 1.5])),
        downsample=tuple(cfg.get("downsample_range", [0.5, 0.9])),
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples")

    model = build_model(model_name, cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model '{model_name}' - {param_count / 1e6:.2f} M trainable params")

    loss_fn = CombinedLoss(
        w_l1=cfg.get("loss_w_l1", 0.5),
        w_ssim=cfg.get("loss_w_ssim", 0.3),
        w_perceptual=cfg.get("loss_w_perceptual", 0.2),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.get("lr", 2e-4),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    num_epochs = cfg.get("epochs", 100)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=cfg.get("lr_min", 1e-6),
    )

    scaler = GradScaler(enabled=(cfg.get("amp", True) and device.type == "cuda"))

    start_epoch = 1
    best_psnr = 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        logger.info(f"Resumed from '{args.resume}' - starting at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch, cfg, logger
        )
        val_metrics = validate(model, val_loader, loss_fn, device, cfg)
        scheduler.step()

        elapsed = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[Epoch {epoch:03d}/{num_epochs}] "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"PSNR={val_metrics['psnr']:.2f} dB  "
            f"SSIM={val_metrics['ssim']:.4f}  "
            f"lr={lr_now:.6f}  "
            f"time={elapsed:.1f}s"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "config": cfg,
        }

        save_checkpoint(state, out_dir / "last.pth")

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            state["best_psnr"] = best_psnr
            save_checkpoint(state, out_dir / "best.pth")
            logger.info(f"  New best PSNR: {best_psnr:.2f} dB - checkpoint saved.")

    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f} dB")
    logger.info(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
