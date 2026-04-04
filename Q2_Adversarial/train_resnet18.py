"""
train_resnet18.py - Train ResNet-18 from scratch on CIFAR-10 (target ≥72% test accuracy).

Usage:
  python train_resnet18.py --epochs 50
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
import wandb

WEIGHTS_DIR   = Path("../weights")
WANDB_PROJECT = "dlops-ass5-q2"
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)


def get_loaders(data_dir="./data", batch_size=128):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    val_n  = 5000
    tr_n   = len(full_train) - val_n
    train_set, val_set = random_split(
        full_train, [tr_n, val_n], generator=torch.Generator().manual_seed(42))

    kw = dict(num_workers=4, pin_memory=True)
    return (DataLoader(train_set, batch_size, True,  **kw),
            DataLoader(val_set,   batch_size, False, **kw),
            DataLoader(test_set,  batch_size, False, **kw))


def build_resnet18_cifar10():
    """Standard ResNet-18 adapted for 32x32 CIFAR images."""
    model = resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--data_dir",   type=str,   default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_resnet18_cifar10().to(device)
    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(project=WANDB_PROJECT, name="resnet18_clean", config=vars(args))

    best_val_acc = 0.0
    ckpt = WEIGHTS_DIR / "Q2_resnet18_clean.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += x.size(0)

        train_loss, train_acc = tl / tt, tc / tt
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | "
              f"Train {train_loss:.4f}/{train_acc*100:.2f}%  "
              f"Val {val_loss:.4f}/{val_acc*100:.2f}%")
        wandb.log(dict(epoch=epoch,
                       train_loss=train_loss, train_acc=train_acc*100,
                       val_loss=val_loss,     val_acc=val_acc*100,
                       lr=optimizer.param_groups[0]['lr']))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    # Final test
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    wandb.log({"test_accuracy": test_acc * 100})
    wandb.finish()

    status = "✔ Target met!" if test_acc >= 0.72 else "✘ Below 72% — train longer!"
    print(status)


if __name__ == "__main__":
    main()
