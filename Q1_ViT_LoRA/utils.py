"""
utils.py - Training helpers, metric logging, LoRA gradient norm tracking.

KEY FIX: gradient norms are captured INSIDE the backward pass,
before optimizer.step() zeros them.
"""
import torch
import torch.nn as nn
import wandb
from typing import Dict, Optional, Tuple


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    log_grad_norms: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (train_loss, train_acc, avg_lora_grad_norms).
    Gradient norms are sampled after loss.backward() and averaged over batches.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    norm_sum:   Dict[str, float] = {}
    norm_count: Dict[str, int]   = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)          # unscale before norm clip
        else:
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()

        # ── Capture LoRA grad norms BEFORE step (gradients still exist) ──
        if log_grad_norms:
            for name, param in model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    n = param.grad.detach().norm().item()
                    norm_sum[name]   = norm_sum.get(name, 0.0) + n
                    norm_count[name] = norm_count.get(name, 0)   + 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)

    avg_norms = {k: norm_sum[k] / norm_count[k] for k in norm_sum}
    return total_loss / total, correct / total, avg_norms


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        loss = criterion(out, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─── WandB Logging ───────────────────────────────────────────────────────────

def log_epoch(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    lr: float,
    lora_grad_norms: Optional[Dict[str, float]] = None,
) -> None:
    metrics: Dict = {
        "epoch":          epoch,
        "train/loss":     train_loss,
        "val/loss":       val_loss,
        "train/accuracy": train_acc * 100,
        "val/accuracy":   val_acc  * 100,
        "lr":             lr,
    }
    # LoRA gradient update graphs (one curve per weight matrix)
    if lora_grad_norms:
        for k, v in lora_grad_norms.items():
            metrics[f"grad_norm/{k}"] = v

    wandb.log(metrics)
    print(
        f"  Epoch {epoch:3d} | "
        f"Train Loss {train_loss:.4f}  Acc {train_acc*100:.2f}% | "
        f"Val   Loss {val_loss:.4f}  Acc {val_acc*100:.2f}%"
    )
