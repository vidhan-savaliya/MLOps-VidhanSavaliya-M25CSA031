"""
train.py - ViT-S fine-tuning on CIFAR-100 (baseline + all LoRA combinations).

Usage:
  python train.py --mode all                          # all 10 experiments
  python train.py --mode baseline                     # head-only fine-tune
  python train.py --mode lora --rank 8 --alpha 8      # single LoRA config
"""
import argparse
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from dataset import get_dataloaders
from model import build_baseline_model, build_lora_model, count_trainable_params
from utils import train_one_epoch, evaluate, log_epoch

# ── Constants ────────────────────────────────────────────────────────────────
WANDB_PROJECT = "dlops-ass5-q1"
WEIGHTS_DIR   = Path("../weights")
NUM_CLASSES   = 100
NUM_EPOCHS    = 10
BATCH_SIZE    = 64
LR            = 1e-3
LORA_RANKS    = [2, 4, 8]
LORA_ALPHAS   = [2, 4, 8]
LORA_DROPOUT  = 0.1


# ── Single experiment ────────────────────────────────────────────────────────

def run_experiment(
    mode: str,
    rank: int       = None,
    alpha: int      = None,
    dropout: float  = LORA_DROPOUT,
    num_epochs: int = NUM_EPOCHS,
    data_dir: str   = "./data",
    device: torch.device = None,
    exp_id: int     = 0,
) -> float:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build model ──────────────────────────────────────────────────────────
    if mode == "baseline":
        model     = build_baseline_model(num_classes=NUM_CLASSES)
        run_name  = "baseline_head_only"
        ckpt_path = WEIGHTS_DIR / "Q1_baseline.pth"
        is_lora   = False
        lora_dir  = None
    else:
        model     = build_lora_model(
                        num_classes=NUM_CLASSES,
                        rank=rank, lora_alpha=alpha, lora_dropout=dropout)
        run_name  = f"lora_r{rank}_a{alpha}_d{dropout}"
        ckpt_path = WEIGHTS_DIR / f"Q1_lora_r{rank}_a{alpha}.pth"
        lora_dir  = WEIGHTS_DIR / f"Q1_lora_r{rank}_a{alpha}_best"
        is_lora   = True

    model = model.to(device)
    n_trainable = count_trainable_params(model)

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=dict(
            mode=mode, rank=rank, alpha=alpha, dropout=dropout,
            epochs=num_epochs, batch_size=BATCH_SIZE, lr=LR,
            trainable_params=n_trainable, exp_id=exp_id,
        ),
        reinit=True,
    )

    # ── Data & optimiser ─────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(data_dir=data_dir,
                                                   batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0

    print(f"\n{'='*65}")
    print(f"Exp {exp_id:2d}: {run_name}  |  trainable={n_trainable:,}  |  {device}")
    print(f"{'='*65}")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):

        # train_one_epoch now returns (loss, acc, grad_norms_dict)
        train_loss, train_acc, grad_norms = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            log_grad_norms=is_lora,   # only track for LoRA runs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_epoch(epoch, train_loss, val_loss, train_acc, val_acc,
                  optimizer.param_groups[0]["lr"],
                  lora_grad_norms=grad_norms if is_lora else None)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if is_lora:
                model.save_pretrained(str(lora_dir))
            else:
                torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ Saved best checkpoint  (val acc = {best_val_acc*100:.2f}%)")

    wandb.finish()
    return best_val_acc


# ── Run all experiments ───────────────────────────────────────────────────────

def run_all(data_dir: str = "./data"):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # 0 — baseline
    acc = run_experiment("baseline", device=device, data_dir=data_dir, exp_id=0)
    results.append(dict(exp=0, mode="baseline", rank="-",
                        alpha="-", dropout="-", acc=acc))

    # 1-9 — all LoRA combinations
    for i, (r, a) in enumerate(product(LORA_RANKS, LORA_ALPHAS), start=1):
        acc = run_experiment("lora", rank=r, alpha=a, dropout=LORA_DROPOUT,
                             device=device, data_dir=data_dir, exp_id=i)
        results.append(dict(exp=i, mode="lora", rank=r,
                            alpha=a, dropout=LORA_DROPOUT, acc=acc))

    # Summary table
    print(f"\n{'='*70}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Exp':>4}  {'Mode':<10} {'Rank':>5} {'Alpha':>6} "
          f"{'Drop':>6} {'ValAcc%':>9}")
    print("-" * 44)
    for r in results:
        print(f"{r['exp']:>4}  {r['mode']:<10} {str(r['rank']):>5} "
              f"{str(r['alpha']):>6} {str(r['dropout']):>6} "
              f"{r['acc']*100:>8.2f}%")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q1: ViT-S LoRA on CIFAR-100")
    parser.add_argument("--mode",     type=str,   choices=["baseline","lora","all"],
                        default="all")
    parser.add_argument("--rank",     type=int,   default=8)
    parser.add_argument("--alpha",    type=int,   default=8)
    parser.add_argument("--dropout",  type=float, default=LORA_DROPOUT)
    parser.add_argument("--epochs",   type=int,   default=NUM_EPOCHS)
    parser.add_argument("--data_dir", type=str,   default="./data")
    parser.add_argument("--exp_id",   type=int,   default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "all":
        run_all(data_dir=args.data_dir)
    elif args.mode == "baseline":
        run_experiment("baseline", device=device,
                       num_epochs=args.epochs, data_dir=args.data_dir)
    else:
        run_experiment("lora",
                       rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                       num_epochs=args.epochs, data_dir=args.data_dir,
                       device=device, exp_id=args.exp_id)


if __name__ == "__main__":
    main()
