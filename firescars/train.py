"""Training entrypoint for single-GPU and multi-GPU (DDP) fine-tuning."""

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from firescars.checkpoint import load_checkpoint, save_checkpoint
from firescars.dataset import FirescarDataset, load_band_stats
from firescars.evaluate import evaluate
from firescars.loss import FirescarLoss
from firescars.model import FirescarModel


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Handle _base_ inheritance
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base_"))
        base = load_config(base_path)
        _deep_update(base, cfg)
        return base
    return cfg


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_steps = cfg["training"]["warmup_steps"]
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=cfg["training"]["min_lr"]
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dist_cfg = cfg.get("distributed", {})
    is_distributed = dist_cfg.get("enabled", False)

    # DDP setup
    rank = 0
    world_size = 1
    local_rank = 0
    if is_distributed:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group(backend=dist_cfg.get("backend", "nccl"))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # Model
    model = FirescarModel(
        encoder_name=cfg["model"].get("encoder", "vit_base_patch16_224"),
        in_chans=cfg["model"].get("input_bands", 6),
        img_size=cfg["model"].get("input_size", 224),
        pretrained_encoder=True,
        encoder_weights_path=cfg["model"].get("encoder_weights_path"),
    ).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Dataset
    band_stats = None
    metadata_path = cfg["data"].get("metadata_path")
    if metadata_path and os.path.exists(metadata_path):
        band_stats = load_band_stats(metadata_path)

    train_ds = FirescarDataset(
        cfg["data"]["chips_path"], split="train", band_stats=band_stats, augment=True
    )
    val_ds = FirescarDataset(
        cfg["data"]["chips_path"], split="val", band_stats=band_stats, augment=False
    )

    # Samplers
    train_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    batch_size = cfg["data"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    # Optimizer with encoder LR multiplier
    base_lr = cfg["training"]["lr"]
    if is_distributed:
        base_lr *= world_size  # linear scaling

    raw_model = model.module if is_distributed else model
    param_groups = raw_model.get_param_groups(cfg["training"]["encoder_lr_multiplier"])
    optimizer = AdamW(
        [
            {"params": param_groups[0]["params"], "lr": base_lr * param_groups[0]["lr_mult"]},
            {"params": param_groups[1]["params"], "lr": base_lr * param_groups[1]["lr_mult"]},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = FirescarLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_path = cfg["checkpoint"]["path"]
    if cfg["checkpoint"].get("resume", True):
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            raw_model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if is_main:
                print(f"Resumed from epoch {start_epoch}")

    # Training loop
    patience_counter = 0
    patience = cfg["training"]["early_stopping"]["patience"]

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(raw_model, val_loader, criterion, device)

        if is_main:
            print(
                f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"IoU={val_metrics['iou']:.4f} | F1={val_metrics['f1']:.4f}"
            )

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if cfg["checkpoint"].get("save_every_epoch", True):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                    is_best=is_best,
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if is_distributed:
        torch.distributed.destroy_process_group()

    if is_main:
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
