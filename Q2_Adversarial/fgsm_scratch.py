"""
fgsm_scratch.py - FGSM attack implemented from scratch (no ART).

Usage:
  python fgsm_scratch.py --checkpoint ../weights/Q2_resnet18_clean.pth
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import wandb

from train_resnet18 import build_resnet18_cifar10, CIFAR10_MEAN, CIFAR10_STD

WEIGHTS_DIR   = Path("../weights")
WANDB_PROJECT = "dlops-ass5-q2"
EPSILON_LIST  = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]


# ── FGSM (from scratch) ──────────────────────────────────────────────────────

def fgsm_attack(model: nn.Module,
                images: torch.Tensor,
                labels: torch.Tensor,
                epsilon: float,
                criterion: nn.Module) -> torch.Tensor:
    """
    x_adv = x + ε · sign(∇_x L(f(x), y))
    Implemented purely with PyTorch autograd — no external library.
    """
    images = images.clone().requires_grad_(True)
    loss   = criterion(model(images), labels)
    model.zero_grad()
    loss.backward()

    adv = images.data + epsilon * images.grad.data.sign()
    # Clamp in normalised space: approx [-2.5, 2.5] covers CIFAR-10 range
    adv = torch.clamp(adv, -3.0, 3.0)
    return adv.detach()


# ── Evaluation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(model, images, labels):
    return (model(images).argmax(1) == labels).float().mean().item()


def evaluate_eps(model, loader, epsilon, criterion, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        x_adv   = fgsm_attack(model, x, y, epsilon, criterion)
        correct += (model(x_adv).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="../weights/Q2_resnet18_clean.pth")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_resnet18_cifar10().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set    = torchvision.datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_set, args.batch_size, False,
                             num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    wandb.init(project=WANDB_PROJECT, name="fgsm_scratch")

    print(f"\n{'Epsilon':>10} | {'Accuracy':>10}")
    print("-" * 25)
    results = []
    for eps in EPSILON_LIST:
        acc = evaluate_eps(model, test_loader, eps, criterion, device)
        results.append((eps, acc))
        print(f"{eps:>10.3f} | {acc*100:>9.2f}%")
        wandb.log({"epsilon": eps, "fgsm_scratch/accuracy": acc * 100})

    wandb.finish()
    return results


if __name__ == "__main__":
    main()
