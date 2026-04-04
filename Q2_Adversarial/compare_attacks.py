"""
compare_attacks.py - Visual comparison of Original vs FGSM-scratch vs FGSM-ART.
Also plots perturbation strength vs accuracy drop and logs everything to WandB.

Usage:
  python compare_attacks.py --checkpoint ../weights/Q2_resnet18_clean.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from train_resnet18 import build_resnet18_cifar10, CIFAR10_MEAN, CIFAR10_STD
from fgsm_scratch   import fgsm_attack, evaluate_eps

WANDB_PROJECT = "dlops-ass5-q2"
EPSILON_LIST  = [0.0, 0.03, 0.1, 0.3]  # Reduced for faster run
CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']


class NormWrapper(nn.Module):
    def __init__(self, base, mean, std, device):
        super().__init__()
        self.base = base
        m = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1).to(device)
        s = torch.tensor(std,  dtype=torch.float32).view(1,3,1,1).to(device)
        self.register_buffer("mean", m)
        self.register_buffer("std",  s)
    def forward(self, x):
        return self.base((x - self.mean) / self.std)


def unnorm(t, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Reverse normalisation for visualisation (CHW tensor → HWC numpy)."""
    m = np.array(mean).reshape(3,1,1)
    s = np.array(std).reshape(3,1,1)
    return np.clip((t.cpu().numpy() * s + m).transpose(1,2,0), 0, 1)


def build_art_classifier(model, device):
    wrapped   = NormWrapper(model, CIFAR10_MEAN, CIFAR10_STD, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    return PyTorchClassifier(
        model=wrapped, loss=criterion, optimizer=optimizer,
        input_shape=(3,32,32), nb_classes=10, clip_values=(0.,1.),
        device_type="gpu" if device.type=="cuda" else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./weights/Q2_resnet18_clean.pth")
    parser.add_argument("--data_dir",   type=str, default="./Q2_Adversarial/data")
    parser.add_argument("--eps",        type=float, default=0.03)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_resnet18_cifar10().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    art_clf = build_art_classifier(model, device)
    criterion = nn.CrossEntropyLoss()

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set    = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_set, 256, False, num_workers=4)

    wandb.init(project=WANDB_PROJECT, name="compare_fgsm_attacks")

    # ── Per-epsilon accuracy table ──────────────────────────────────────────
    print(f"\n{'ε':>6} | {'Clean':>8} | {'FGSM-scratch':>13} | {'FGSM-ART':>9}")
    print("-" * 45)

    acc_clean_list, acc_scratch_list, acc_art_list = [], [], []

    mean_np = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1,3,1,1)
    std_np  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1,3,1,1)

    for eps in EPSILON_LIST:
        acc_scratch = evaluate_eps(model, test_loader, eps, criterion, device)

        # ART evaluation
        xs, ys = [], []
        for x, y in test_loader:
            x_np = x.numpy() * std_np + mean_np
            xs.append(np.clip(x_np, 0, 1))
            ys.append(y.numpy())
        X_np = np.concatenate(xs)
        Y_np = np.concatenate(ys)

        if eps == 0.0:
            preds_art = art_clf.predict(X_np).argmax(1)
        else:
            X_adv     = FastGradientMethod(art_clf, eps=eps).generate(X_np)
            preds_art = art_clf.predict(X_adv).argmax(1)
        acc_art = (preds_art == Y_np).mean()

        acc_clean_list.append(acc_scratch if eps == 0.0 else acc_clean_list[0])
        acc_scratch_list.append(acc_scratch)
        acc_art_list.append(acc_art)

        print(f"{eps:>6.3f} | {acc_clean_list[-1]*100:>7.2f}% | "
              f"{acc_scratch*100:>12.2f}% | {acc_art*100:>8.2f}%")
        wandb.log({"epsilon": eps,
                   "compare/acc_scratch": acc_scratch*100,
                   "compare/acc_art":     acc_art*100})

    # ── Plot perturbation vs accuracy ───────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(EPSILON_LIST, [a*100 for a in acc_scratch_list], "o-b", label="FGSM Scratch")
    plt.plot(EPSILON_LIST, [a*100 for a in acc_art_list],     "s--r", label="FGSM ART")
    plt.xlabel("Epsilon (perturbation strength)")
    plt.ylabel("Accuracy (%)")
    plt.title("Perturbation Strength vs Accuracy Drop")
    plt.legend(); plt.grid(alpha=0.3)
    Path("./plots").mkdir(exist_ok=True)
    plt.savefig("./plots/eps_vs_accuracy.png", dpi=150)
    plt.close()
    wandb.log({"compare/eps_vs_accuracy": wandb.Image("./plots/eps_vs_accuracy.png")})

    # ── Visual comparison: 10 sample images ────────────────────────────────
    sample_loader = DataLoader(test_set, 10, False)
    imgs, labs    = next(iter(sample_loader))
    imgs, labs    = imgs.to(device), labs.to(device)
    eps_vis       = args.eps

    adv_scratch = fgsm_attack(model, imgs, labs, eps_vis, criterion)
    x_np   = imgs.cpu().numpy() * std_np + mean_np
    x_np   = np.clip(x_np, 0, 1)
    X_art  = FastGradientMethod(art_clf, eps=eps_vis).generate(x_np)

    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    row_labs  = ["Original", f"FGSM Scratch ε={eps_vis}", f"FGSM ART ε={eps_vis}"]

    for col in range(10):
        axes[0, col].imshow(unnorm(imgs[col]))
        axes[1, col].imshow(unnorm(adv_scratch[col]))
        axes[2, col].imshow(X_art[col].transpose(1,2,0))
        for row in range(3):
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_labs[row], fontsize=8)

    axes[0, 0].set_title(CIFAR10_LABELS[labs[0]], fontsize=7)
    plt.suptitle(f"Original vs FGSM Attacks (ε={eps_vis})", fontsize=12)
    plt.tight_layout()
    plt.savefig("./plots/fgsm_comparison.png", dpi=150)
    plt.close()
    wandb.log({"compare/visual_comparison": wandb.Image("./plots/fgsm_comparison.png")})

    wandb.finish()
    print("\nPlots saved to ./plots/")


if __name__ == "__main__":
    main()
