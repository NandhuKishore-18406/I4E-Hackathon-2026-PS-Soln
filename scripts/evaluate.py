import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_loader import build_test_loader, PairedImageDataset
from metrics.eval_metrics import MetricEvaluator


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_cfg = ckpt.get("config", cfg)
    model_name = saved_cfg.get("model", cfg.get("model", "swinir")).lower()

    if model_name == "swinir":
        from models.swinir_model import build_swinir
        model = build_swinir(
            variant=saved_cfg.get("swinir_variant", "small"),
            in_channels=saved_cfg.get("in_channels", 3),
            out_channels=saved_cfg.get("out_channels", 3),
            window_size=saved_cfg.get("window_size", 8),
            upscale=saved_cfg.get("upscale", 1),
        )
    elif model_name == "unet":
        from models.unet_model import UNet
        model = UNet(
            in_channels=saved_cfg.get("in_channels", 3),
            out_channels=saved_cfg.get("out_channels", 3),
        )
    else:
        raise ValueError(f"Unknown model '{model_name}' in checkpoint config.")

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, model_name


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0.0, 1.0).cpu()
    return TF.to_pil_image(t)


def save_images(
    batch: dict,
    pred: torch.Tensor,
    out_dir: Path,
    save_input: bool = True,
    save_target: bool = True,
):
    for i in range(pred.shape[0]):
        stem = Path(batch["input_path"][i]).stem
        tensor_to_pil(pred[i]).save(out_dir / f"{stem}_pred.png")
        if save_input:
            tensor_to_pil(batch["input"][i]).save(out_dir / f"{stem}_input.png")
        if save_target and "target" in batch:
            tensor_to_pil(batch["target"][i]).save(out_dir / f"{stem}_target.png")


def write_report(metrics: dict, per_image: list, out_dir: Path):
    report_path = out_dir / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("  Image Restoration - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Metric':<15} {'Value':>10}\n")
        f.write("-" * 27 + "\n")
        f.write(f"{'PSNR (dB)':<15} {metrics['psnr']:>10.4f}\n")
        f.write(f"{'SSIM':<15} {metrics['ssim']:>10.4f}\n")
        f.write(f"\nPer-image results ({len(per_image)} images):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Filename':<35} {'PSNR':>8} {'SSIM':>8}\n")
        f.write("-" * 50 + "\n")
        for entry in per_image:
            f.write(f"{entry['name']:<35} {entry['psnr']:>8.2f} {entry['ssim']:>8.4f}\n")
    return report_path


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    device: torch.device,
    out_dir: Path,
    compute_metrics: bool,
    amp_enabled: bool,
    logger: logging.Logger,
) -> dict:
    evaluator = MetricEvaluator()
    per_image_results = []
    t0 = time.time()

    for i, batch in enumerate(loader):
        inp = batch["input"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            pred = model(inp)

        pred = pred.clamp(0.0, 1.0).float()

        if compute_metrics and "target" in batch:
            tgt = batch["target"].to(device, non_blocking=True)
            evaluator.update(pred, tgt)

            from metrics.eval_metrics import psnr_batch, ssim_batch
            psnr_vals = psnr_batch(pred, tgt)
            ssim_vals = ssim_batch(pred, tgt)
            for j in range(pred.shape[0]):
                name = Path(batch["input_path"][j]).stem
                per_image_results.append(
                    {"name": name, "psnr": psnr_vals[j], "ssim": ssim_vals[j]}
                )

        save_images(
            batch=batch,
            pred=pred,
            out_dir=out_dir,
            save_input=True,
            save_target=compute_metrics,
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(loader)} batches...")

    elapsed = time.time() - t0
    logger.info(f"Inference complete - {len(loader)} images in {elapsed:.1f}s")

    metrics = evaluator.compute() if compute_metrics else {}
    return metrics, per_image_results


def parse_args():
    parser = argparse.ArgumentParser(description="Image Restoration Evaluation")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt_stem = Path(args.checkpoint).parent.name
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / cfg.get("output_dir", "outputs/test_outputs") / ckpt_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to: {out_dir}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, model_name = load_model(args.checkpoint, cfg, device)
    logger.info(f"Model '{model_name}' loaded successfully.")

    data_root = cfg.get("data_root", "data")
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    test_loader = build_test_loader(
        data_root=data_root,
        num_workers=cfg.get("num_workers", 2),
    )
    logger.info(f"Test set: {len(test_loader.dataset)} images")

    compute_metrics = not args.no_metrics
    amp_enabled = cfg.get("amp", True) and device.type == "cuda"

    metrics, per_image = run_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=out_dir,
        compute_metrics=compute_metrics,
        amp_enabled=amp_enabled,
        logger=logger,
    )

    if compute_metrics and metrics:
        logger.info(
            f"Results - PSNR: {metrics['psnr']:.4f} dB  |  SSIM: {metrics['ssim']:.4f}"
        )
        report_path = write_report(metrics, per_image, out_dir)
        logger.info(f"Report written to: {report_path}")
    else:
        logger.info("Metric computation skipped (--no-metrics).")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
