#!/usr/bin/env python3
"""
Main training script for City_Scape semantic segmentation.

Optimizations for NVIDIA RTX PRO 6000 Blackwell (sm_120):
- channels_last memory format (Tensor Cores friendly, +15-22% on conv)
- TF32 matmul precision ('high')
- cudnn.benchmark for fixed-shape input auto-tuning
- BF16 mixed precision (Blackwell native, no NaN)
- Versioned checkpoints every cfg.evaluation.checkpoint_every_epochs
- Auto-resume from checkpoints/<run>/last.pth (full state)
- Inline validation OFF by default — use scripts/evaluate.py offline

Usage:
    taskset -c 0-15 python scripts/train.py +experiment=pilot_fullres_B_baseline
    # Extend training:
    taskset -c 0-15 python scripts/train.py +experiment=pilot_fullres_B_baseline training.epochs=200
"""

import os
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.models import build_model
from src.losses import build_loss
from src.data import build_dataloaders
from src.metrics import SegmentationMetrics


def _setup_blackwell_optims(deterministic: bool):
    """Configure global PyTorch flags for Blackwell sm_120."""
    # TF32 on matmul -- Blackwell native, free speedup vs FP32
    torch.set_float32_matmul_precision("high")
    # cuDNN auto-tune for fixed input shapes; conflicts with strict determinism
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _wandb_log(log: dict):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(log)
    except Exception:
        pass


def _wandb_finish():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps, channels_last):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sdt = batch["sdt"].to(device, non_blocking=True) if "sdt" in batch else None
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            if isinstance(output, tuple):
                main_out, aux_out = output
                loss_main, _ = criterion(main_out, labels, sdt=sdt)
                loss_aux, _ = criterion(aux_out, labels, sdt=sdt)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss, _ = criterion(output, labels, sdt=sdt)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _save_checkpoint(path: Path, *, epoch, model, optimizer, scheduler, scaler, best_miou, cfg):
    """Atomic full-state save with fsync."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    rng = ckpt.get("rng", {})
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            cuda_states = [s.cpu().to(torch.uint8) for s in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception as e:
        print(f"  [resume] RNG state restore skipped: {type(e).__name__}: {e}")
    return ckpt["epoch"] + 1, ckpt.get("best_miou", 0.0)


def _build_scheduler(optimizer, cfg, total_epochs, last_epoch=-1):
    return torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=total_epochs,
        power=cfg.training.scheduler.get("power", 1.0),
        last_epoch=last_epoch,
    )


def train_one_seed(cfg: DictConfig, seed: int, device: torch.device):
    print(f"\n{'='*80}\nTraining with seed {seed}\n{'='*80}")
    deterministic = cfg.training.get("deterministic", False)
    set_seed(seed, deterministic)
    _setup_blackwell_optims(deterministic)

    channels_last = cfg.training.get("channels_last", True)
    use_grad_checkpoint = cfg.training.get("gradient_checkpointing", False)

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = build_loss(cfg, class_frequencies=dataloaders["class_frequencies"]).to(device)

    if use_grad_checkpoint and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)

    opt_cfg = cfg.training.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
    )

    total_epochs = cfg.training.epochs
    scheduler = _build_scheduler(optimizer, cfg, total_epochs)
    scaler = torch.amp.GradScaler("cuda")

    run_name = f"{cfg.experiment.name}_seed{seed}"
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pth"

    start_epoch = 0
    best_miou = 0.0
    if last_path.exists():
        start_epoch, best_miou = _load_checkpoint(
            last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        if scheduler.total_iters != total_epochs:
            print(f"  [resume] training.epochs changed ({scheduler.total_iters} -> {total_epochs}), rebuilding scheduler")
            # Reset optimizer lr to initial before rebuild — PolyLR is multiplicative,
            # if previous run ended with lr=0 the new scheduler stays at 0 forever.
            init_lr = float(opt_cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = init_lr
                g["initial_lr"] = init_lr
            scheduler = _build_scheduler(optimizer, cfg, total_epochs, last_epoch=start_epoch - 1)
            # Manually advance to compute correct lr at resumed epoch
            for g, lr_now in zip(optimizer.param_groups, scheduler.get_last_lr()):
                g["lr"] = lr_now
            print(f"  [resume] reset optimizer lr to {init_lr:.2e}, scheduler at last_epoch={start_epoch-1}, current lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"  [resume] epoch {start_epoch}/{total_epochs}, best_miou={best_miou:.4f}")
        if start_epoch >= total_epochs:
            print(f"  [resume] already complete, skipping seed {seed}")
            return best_miou

    try:
        import wandb
        wandb.init(
            project=cfg.experiment.wandb_project,
            name=run_name, id=run_name, resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.experiment.get("wandb_tags", [])),
        )
    except Exception as e:
        print(f"  wandb init failed ({e}), continuing without wandb")

    accum_steps = cfg.training.get("gradient_accumulation_steps", 1)
    ckpt_every = cfg.evaluation.get("checkpoint_every_epochs", 2)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scaler, device, accum_steps, channels_last
        )
        scheduler.step()
        elapsed = time.time() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "vram_peak_gb": vram_gb,
            "epoch_time_s": elapsed,
        }
        print(
            f"Epoch {epoch+1}/{total_epochs} | train_loss={train_loss:.4f} | "
            f"lr={log['lr']:.2e} | VRAM={vram_gb:.1f}GB | time={elapsed:.0f}s"
        )

        # last.pth: every epoch (crash recovery, overwrite)
        _save_checkpoint(
            last_path, epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, best_miou=best_miou, cfg=cfg,
        )

        # epoch_N.pth: versioned snapshot every K epochs (kept for offline eval)
        is_last = epoch == total_epochs - 1
        if (epoch + 1) % ckpt_every == 0 or is_last:
            versioned = checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
            _save_checkpoint(
                versioned, epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, best_miou=best_miou, cfg=cfg,
            )
            print(f"  [ckpt] {versioned.name} written")

        _wandb_log(log)

    print(f"\nSeed {seed} done after {total_epochs} epochs.")
    _wandb_finish()
    return best_miou


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.experiment.get("seeds", [42])
    for seed in seeds:
        train_one_seed(cfg, seed, device)


if __name__ == "__main__":
    main()
