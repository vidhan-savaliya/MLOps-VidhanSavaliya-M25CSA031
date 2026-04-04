"""
fgsm_art.py - FGSM attack using IBM Adversarial Robustness Toolbox (ART).

Usage:
  python fgsm_art.py --checkpoint ../weights/Q2_resnet18_clean.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from train_resnet18 import build_resnet18_cifar10, CIFAR10_MEAN, CIFAR10_STD

WANDB_PROJECT = "dlops-ass5-q2"
EPSILON_LIST  = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]


def to_numpy_01(loader, n_batches=None):
    """Return (X, y) as float32 numpy arrays in [0, 1] (unnormalized)."""
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1, 3, 1, 1)

    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        x_np = x.numpy()
        # Reverse normalisation → [0,1]
        x_np = x_np * std + mean
        x_np = np.clip(x_np, 0.0, 1.0)
        xs.append(x_np)
        ys.append(y.numpy())
        if n_batches and i + 1 >= n_batches:
            break
    return np.concatenate(xs), np.concatenate(ys)


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

    # ART needs unnormalized [0,1] input — we handle normalization inside a wrapper
    class NormWrapper(nn.Module):
        """Wraps the model to accept [0,1] input and apply normalization internally."""
        def __init__(self, base, mean, std):
            super().__init__()
            self.base = base
            m = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1).to(device)
            s = torch.tensor(std,  dtype=torch.float32).view(1,3,1,1).to(device)
            self.register_buffer("mean", m)
            self.register_buffer("std",  s)

        def forward(self, x):
            return self.base((x - self.mean) / self.std)

    wrapped = NormWrapper(model, CIFAR10_MEAN, CIFAR10_STD).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=wrapped,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # Load test data (unnormalized [0,1] for ART)
    raw_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set    = torchvision.datasets.CIFAR10(
        args.data_dir, False, True, transform=raw_tf)
    test_loader = DataLoader(test_set, args.batch_size, False, num_workers=4)

    X_test, y_test = to_numpy_01(test_loader)

    wandb.init(project=WANDB_PROJECT, name="fgsm_art")
    print(f"\n{'Epsilon':>10} | {'Accuracy':>10}")
    print("-" * 25)

    for eps in EPSILON_LIST:
        if eps == 0.0:
            preds = classifier.predict(X_test).argmax(1)
        else:
            attack    = FastGradientMethod(estimator=classifier, eps=eps)
            X_adv     = attack.generate(x=X_test)
            preds     = classifier.predict(X_adv).argmax(1)

        acc = (preds == y_test).mean()
        print(f"{eps:>10.3f} | {acc*100:>9.2f}%")
        wandb.log({"epsilon": eps, "fgsm_art/accuracy": acc * 100})

    wandb.finish()


if __name__ == "__main__":
    main()
