"""Evaluation metrics for binary segmentation."""

import torch


def compute_metrics(logits, targets, threshold=0.5):
    """Compute IoU, F1, precision, recall from logits and binary targets."""
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Run evaluation over a dataloader. Returns avg loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = avg_loss
    return metrics
