"""
test.py - Evaluate ViT-S checkpoint on CIFAR-100 test set.

Usage:
  # Baseline:
  python test.py --mode baseline --checkpoint ../weights/Q1_baseline.pth

  # LoRA (PEFT adapter folder):
  python test.py --mode lora --checkpoint ../weights/Q1_lora_r8_a8_best \
                 --rank 8 --alpha 8 --dropout 0.1

  # Run ALL checkpoints and print summary table:
  python test.py --mode all
"""
import os
import argparse
from pathlib import Path
from itertools import product

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from dataset import get_dataloaders, CIFAR100_CLASSES
from model import build_baseline_model, build_lora_model, count_trainable_params

WANDB_PROJECT = "dlops-ass5-q1"
NUM_CLASSES   = 100
WEIGHTS_DIR   = Path(__file__).parent.parent / "weights"
LORA_RANKS    = [2, 4, 8]
LORA_ALPHAS   = [2, 4, 8]
LORA_DROPOUT  = 0.1


# ── Class-wise evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classwise(model, loader, device):
    model.eval()
    correct = np.zeros(NUM_CLASSES)
    total   = np.zeros(NUM_CLASSES)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for c in range(NUM_CLASSES):
            mask       = (labels == c)
            total[c]  += mask.sum().item()
            correct[c]+= (preds[mask] == c).sum().item()

    classwise = correct / (total + 1e-8)
    overall   = correct.sum() / total.sum()
    return classwise, float(overall)


# ── Histogram plot ───────────────────────────────────────────────────────────

def plot_histogram(classwise_acc: np.ndarray, run_name: str,
                   save_dir: str = "./plots") -> str:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(24, 6))
    ax.bar(range(NUM_CLASSES), classwise_acc * 100, color="steelblue", alpha=0.85)
    ax.axhline(classwise_acc.mean() * 100, color="crimson", linestyle="--",
               label=f"Mean {classwise_acc.mean()*100:.1f}%")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Class-wise Test Accuracy — {run_name}", fontsize=13)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=5.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{run_name}_classwise.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── Load model helpers ───────────────────────────────────────────────────────

def load_baseline(ckpt: str, device) -> torch.nn.Module:
    model = build_baseline_model(NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device)


def load_lora(lora_dir: str, rank: int, alpha: int,
              dropout: float, device) -> torch.nn.Module:
    """Load a PEFT LoRA adapter from its saved directory."""
    from peft import PeftModel
    import timm

    # Build and re-wrap with same LoRA config
    model = build_lora_model(
        num_classes=NUM_CLASSES, rank=rank,
        lora_alpha=alpha, lora_dropout=dropout)

    # Load adapter weights (PEFT saves adapter_model.safetensors or .bin)
    adapter_bin = Path(lora_dir) / "adapter_model.bin"
    adapter_st  = Path(lora_dir) / "adapter_model.safetensors"

    if adapter_st.exists():
        from safetensors.torch import load_file
        state = load_file(str(adapter_st))
        model.load_state_dict(state, strict=False)
    elif adapter_bin.exists():
        state = torch.load(str(adapter_bin), map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"No adapter weights in {lora_dir}")

    return model.to(device)


# ── Single evaluation ────────────────────────────────────────────────────────

def run_eval(model, test_loader, device, run_name: str,
             n_trainable: int, meta: dict):
    classwise, overall = evaluate_classwise(model, test_loader, device)
    hist_path = plot_histogram(classwise, run_name)

    print(f"\n[{run_name}] Overall Test Acc: {overall*100:.2f}%  "
          f"Trainable params: {n_trainable:,}")

    wandb.init(project=WANDB_PROJECT, name=f"test_{run_name}", reinit=True)
    data  = [[c, float(a * 100)] for c, a in zip(CIFAR100_CLASSES, classwise)]
    table = wandb.Table(data=data, columns=["class", "accuracy"])
    wandb.log({
        "test/overall_accuracy":    overall * 100,
        "test/trainable_params":    n_trainable,
        "test/classwise_histogram": wandb.Image(hist_path),
        "test/classwise_bar":       wandb.plot.bar(
            table, "class", "accuracy",
            title=f"Class-wise Accuracy — {run_name}"),
        **meta,
    })
    wandb.finish()
    return overall


# ── All-in-one mode ──────────────────────────────────────────────────────────

def run_all_evals(data_dir: str, device: torch.device):
    _, _, test_loader = get_dataloaders(data_dir, batch_size=128)
    summary = []

    # Baseline
    ckpt = WEIGHTS_DIR / "Q1_baseline.pth"
    if ckpt.exists():
        model = load_baseline(str(ckpt), device)
        n_t   = count_trainable_params(model)
        acc   = run_eval(model, test_loader, device, "baseline", n_t,
                         {"lora": False, "rank": None, "alpha": None, "dropout": None})
        summary.append(dict(mode="baseline", rank="-", alpha="-",
                            dropout="-", acc=acc, params=n_t))

    # LoRA runs
    for r, a in product(LORA_RANKS, LORA_ALPHAS):
        lora_dir = WEIGHTS_DIR / f"Q1_lora_r{r}_a{a}_best"
        if not lora_dir.exists():
            print(f"  Skipping r={r} a={a} — checkpoint not found")
            continue
        model = load_lora(str(lora_dir), r, a, LORA_DROPOUT, device)
        n_t   = count_trainable_params(model)
        name  = f"lora_r{r}_a{a}"
        acc   = run_eval(model, test_loader, device, name, n_t,
                         {"lora": True, "rank": r, "alpha": a, "dropout": LORA_DROPOUT})
        summary.append(dict(mode="lora", rank=r, alpha=a,
                            dropout=LORA_DROPOUT, acc=acc, params=n_t))

    # Print summary table
    print(f"\n{'='*72}")
    print("TEST SUMMARY TABLE")
    print(f"{'='*72}")
    print(f"{'Mode':<12} {'Rank':>5} {'Alpha':>6} {'Drop':>6} "
          f"{'TestAcc%':>10} {'Trainable Params':>18}")
    print("-" * 72)
    for r in summary:
        print(f"{r['mode']:<12} {str(r['rank']):>5} {str(r['alpha']):>6} "
              f"{str(r['dropout']):>6} {r['acc']*100:>9.2f}% "
              f"{r['params']:>18,}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str,
                        choices=["baseline", "lora", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rank",       type=int,   default=8)
    parser.add_argument("--alpha",      type=int,   default=8)
    parser.add_argument("--dropout",    type=float, default=LORA_DROPOUT)
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--batch_size", type=int,   default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "all":
        run_all_evals(args.data_dir, device)
        return

    _, _, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    if args.mode == "baseline":
        ckpt  = args.checkpoint or str(WEIGHTS_DIR / "Q1_baseline.pth")
        model = load_baseline(ckpt, device)
        name  = "baseline"
        meta  = {"lora": False}
    else:
        ldir  = args.checkpoint or str(
                    WEIGHTS_DIR / f"Q1_lora_r{args.rank}_a{args.alpha}_best")
        model = load_lora(ldir, args.rank, args.alpha, args.dropout, device)
        name  = f"lora_r{args.rank}_a{args.alpha}"
        meta  = {"lora": True, "rank": args.rank,
                 "alpha": args.alpha, "dropout": args.dropout}

    n_t = count_trainable_params(model)
    run_eval(model, test_loader, device, name, n_t, meta)


if __name__ == "__main__":
    main()
