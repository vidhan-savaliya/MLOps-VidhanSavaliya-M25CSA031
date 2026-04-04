"""
optuna_search.py - Optuna hyperparameter search for LoRA on ViT-S CIFAR-100.

Usage:
  python optuna_search.py --n_trials 30 --data_dir ./data
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import optuna
import wandb

from dataset import get_dataloaders
from model import build_lora_model
from utils import train_one_epoch, evaluate

WANDB_PROJECT = "dlops-ass5-q1-optuna"
WEIGHTS_DIR   = Path("../weights")
NUM_CLASSES   = 100
BATCH_SIZE    = 64
LR            = 1e-3
TRIAL_EPOCHS  = 5     # short run for search
FINAL_EPOCHS  = 10    # full run for best config


# ── Objective ────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, device: torch.device, data_dir: str) -> float:
    rank    = trial.suggest_categorical("rank",    [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.2])

    wandb.init(
        project=WANDB_PROJECT,
        name=f"trial_{trial.number}_r{rank}_a{alpha}_d{dropout}",
        config=dict(trial=trial.number, rank=rank, alpha=alpha, dropout=dropout),
        reinit=True,
    )

    model = build_lora_model(
        num_classes=NUM_CLASSES, rank=rank,
        lora_alpha=alpha, lora_dropout=dropout).to(device)

    train_loader, val_loader, _ = get_dataloaders(data_dir, BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRIAL_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    for epoch in range(1, TRIAL_EPOCHS + 1):
        # Unpack 3-tuple: (loss, acc, grad_norms)
        _, _, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            log_grad_norms=False)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({"epoch": epoch, "val/accuracy": val_acc * 100})
        trial.report(val_acc, epoch)

        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

    wandb.finish()
    return best_val_acc


# ── Retrain best config ──────────────────────────────────────────────────────

def retrain_best(best_params: dict, data_dir: str, device: torch.device):
    r, a, d = best_params["rank"], best_params["alpha"], best_params["dropout"]
    print(f"\nRetraining best config: rank={r}, alpha={a}, dropout={d}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_lora_model(
        num_classes=NUM_CLASSES, rank=r,
        lora_alpha=a, lora_dropout=d).to(device)

    wandb.init(project=WANDB_PROJECT,
               name=f"best_r{r}_a{a}_d{d}",
               config=best_params, reinit=True)

    train_loader, val_loader, _ = get_dataloaders(data_dir, BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    save_dir = WEIGHTS_DIR / f"Q1_optuna_best_r{r}_a{a}"
    best_acc = 0.0

    for epoch in range(1, FINAL_EPOCHS + 1):
        tl, ta, grad_norms = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            log_grad_norms=True)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = {
            "epoch": epoch,
            "train/loss": tl, "train/accuracy": ta * 100,
            "val/loss":   vl, "val/accuracy":   va * 100,
        }
        for k, v in grad_norms.items():
            metrics[f"grad_norm/{k}"] = v
        wandb.log(metrics)

        print(f"  Epoch {epoch:2d} | TrainAcc {ta*100:.2f}%  ValAcc {va*100:.2f}%")

        if va > best_acc:
            best_acc = va
            model.save_pretrained(str(save_dir))
            print(f"  ✔ Saved  (best val acc = {best_acc*100:.2f}%)")

    wandb.finish()
    print(f"\nBest Optuna model saved → {save_dir}")

    # Also save as single best path for push_to_hub
    best_link = WEIGHTS_DIR / "Q1_best_model"
    if not best_link.exists():
        import shutil
        shutil.copytree(str(save_dir), str(best_link))

    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optuna on: {device}")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="vit_lora_search",
    )
    study.optimize(
        lambda t: objective(t, device, args.data_dir),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print(f"Best trial   : {study.best_trial.number}")
    print(f"Best val acc : {study.best_value*100:.2f}%")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    retrain_best(study.best_params, args.data_dir, device)


if __name__ == "__main__":
    main()
