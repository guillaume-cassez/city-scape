#!/usr/bin/env python3
"""
Evaluation script: compute mIoU, per-class IoU, boundary F1, trimap mIoU on val/test.
Decoupled from training — runs offline against any saved checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/pilot_fullres_B_baseline_seed42/epoch_002.pth
    python scripts/evaluate.py --checkpoint .../epoch_010.pth --tta
    # Eval all snapshots:
    for c in checkpoints/pilot_fullres_B_baseline_seed42/epoch_*.pth; do
        python scripts/evaluate.py --checkpoint "$c"
    done
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.models import build_model
from src.data import build_dataloaders
from src.metrics import SegmentationMetrics
from src.metrics.segmentation_metrics import compute_boundary_f1, compute_trimap_iou
from src.data.cityscapes_dataset import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <ckpt>.results.json)")
    parser.add_argument("--no-channels-last", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".results.json")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    channels_last = (not args.no_channels_last) and cfg.training.get("channels_last", True)

    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.tta:
        from src.postprocessing.tta import TTAWrapper
        model = TTAWrapper(model, scales=[0.5, 0.75, 1.0, 1.25, 1.5], flip=True)

    model.eval()

    dataloaders = build_dataloaders(cfg)
    loader = dataloaders[args.split] if args.split in dataloaders else dataloaders["val"]

    metrics = SegmentationMetrics(num_classes=19)
    boundary_f1s = []
    trimap_mious = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]

            metrics.update(output, labels)

            pred_np = output.argmax(dim=1).cpu().numpy()
            label_np = labels.cpu().numpy()
            for b in range(pred_np.shape[0]):
                bf1 = compute_boundary_f1(pred_np[b], label_np[b])
                boundary_f1s.append(bf1["boundary_f1"])
                trimap = compute_trimap_iou(pred_np[b], label_np[b])
                trimap_mious.append(trimap["trimap_mIoU"])

    elapsed = time.time() - t0
    results = metrics.compute()

    record = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)) + 1,  # 1-indexed for readability
        "experiment": cfg.experiment.name,
        "run": ckpt_path.parent.name,
        "split": args.split,
        "tta": bool(args.tta),
        "elapsed_s": round(elapsed, 1),
        "mIoU": float(results["mIoU"]),
        "boundary_f1_mean": float(np.mean(boundary_f1s)),
        "trimap_mIoU_mean": float(np.mean(trimap_mious)),
        "per_class_iou": {CLASS_NAMES[i]: float(results["per_class_iou"][i]) for i in range(19)},
    }
    out_path.write_text(json.dumps(record, indent=2))

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.split})")
    print(f"{'='*60}")
    print(f"mIoU:               {record['mIoU']*100:.2f}%")
    print(f"Boundary F1 (mean): {record['boundary_f1_mean']*100:.2f}%")
    print(f"Trimap mIoU (mean): {record['trimap_mIoU_mean']*100:.2f}%")
    print(f"\n{'Per-class IoU':=^60}")
    for name, iou in record["per_class_iou"].items():
        print(f"  {name:20s}: {iou*100:.2f}%")
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"JSON   : {out_path}")


if __name__ == "__main__":
    main()
