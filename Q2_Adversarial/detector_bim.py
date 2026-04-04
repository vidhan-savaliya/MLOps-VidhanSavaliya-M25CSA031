"""
detector_bim.py - Binary adversarial detector (clean vs BIM-adversarial) using ResNet-34.

Usage:
  python detector_bim.py --clean_ckpt ../weights/Q2_resnet18_clean.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import BasicIterativeMethod

from train_resnet18 import build_resnet18_cifar10, CIFAR10_MEAN, CIFAR10_STD
from detector_pgd   import build_detector, NormWrapper

WEIGHTS_DIR   = Path("../weights")
WANDB_PROJECT = "dlops-ass5-q2"


def generate_bim_examples(victim_model, data, device,
                           eps=0.03, eps_step=0.007, max_iter=20):
    """Use ART BIM (BasicIterativeMethod) to create adversarial examples."""
    wrapped   = NormWrapper(victim_model, CIFAR10_MEAN, CIFAR10_STD, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    clf = PyTorchClassifier(
        model=wrapped, loss=criterion, optimizer=optimizer,
        input_shape=(3,32,32), nb_classes=10, clip_values=(0.,1.),
        device_type="gpu" if device.type=="cuda" else "cpu")

    attack = BasicIterativeMethod(
        estimator=clf, eps=eps, eps_step=eps_step, max_iter=max_iter)
    return attack.generate(x=data)


def build_detection_dataset(victim_model, data_dir, device, batch_size=256):
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set    = torchvision.datasets.CIFAR10(data_dir, False, True, transform=test_tf)
    test_loader = DataLoader(test_set, batch_size, False, num_workers=4)

    mean_np = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1,3,1,1)
    std_np  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1,3,1,1)

    xs_clean, xs_adv = [], []
    for x, _ in test_loader:
        x_np = np.clip(x.numpy() * std_np + mean_np, 0, 1)
        xs_clean.append(x_np)
        xs_adv.append(generate_bim_examples(victim_model, x_np, device))

    X_clean = np.concatenate(xs_clean)
    X_adv   = np.concatenate(xs_adv)

    # Re-normalise for ImageNet-pretrained ResNet-34
    in_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
    in_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)
    X_clean_n = (X_clean - in_mean) / in_std
    X_adv_n   = (X_adv   - in_mean) / in_std

    X = np.concatenate([X_clean_n, X_adv_n]).astype(np.float32)
    y = np.array([0]*len(X_clean_n) + [1]*len(X_adv_n), dtype=np.int64)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    n_val   = len(dataset) // 5
    return random_split(dataset, [len(dataset)-n_val, n_val],
                        generator=torch.Generator().manual_seed(42))


def train_detector(model, train_ds, val_ds, device, epochs=20):
    train_ld = DataLoader(train_ds, 128, True,  num_workers=2)
    val_ld   = DataLoader(val_ds,   128, False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    ckpt     = WEIGHTS_DIR / "Q2_detector_bim.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x);  loss = criterion(out, y)
            loss.backward();  optimizer.step()
            tl += loss.item()*x.size(0)
            tc += (out.argmax(1)==y).sum().item()
            tt += x.size(0)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1)==y).sum().item()
                vt += x.size(0)

        val_acc = vc / vt
        scheduler.step()
        print(f"Epoch {epoch:3d} | Train {tc/tt*100:.2f}%  Val {val_acc*100:.2f}%")
        wandb.log({"epoch": epoch,
                   "bim_detector/train_acc": tc/tt*100,
                   "bim_detector/val_acc":   val_acc*100})

        if val_acc > best_acc:
            best_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    print(f"\nBest detector accuracy (BIM): {best_acc*100:.2f}%")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_ckpt", type=str,
                        default="../weights/Q2_resnet18_clean.pth")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--epochs",     type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    victim = build_resnet18_cifar10().to(device)
    victim.load_state_dict(torch.load(args.clean_ckpt, map_location=device))
    victim.eval()

    print("Generating BIM adversarial examples...")
    train_ds, val_ds = build_detection_dataset(victim, args.data_dir, device)

    detector = build_detector().to(device)
    wandb.init(project=WANDB_PROJECT, name="detector_bim",
               config={"epochs": args.epochs, "attack": "BIM"})
    best_acc = train_detector(detector, train_ds, val_ds, device, args.epochs)
    wandb.log({"bim_detector/best_val_acc": best_acc * 100})
    wandb.finish()

    status = "✔ Target met!" if best_acc >= 0.70 else "✘ Below 70%"
    print(f"Detection accuracy: {best_acc*100:.2f}%  {status}")


if __name__ == "__main__":
    main()
