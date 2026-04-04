"""
wandb_visualize.py - Upload 10 sample clean + adversarial images for all four
attack types (FGSM-scratch, FGSM-ART, PGD, BIM) to WandB.

Usage:
  python wandb_visualize.py --clean_ckpt ../weights/Q2_resnet18_clean.pth
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
from art.attacks.evasion import (FastGradientMethod,
                                  ProjectedGradientDescentPyTorch,
                                  BasicIterativeMethod)

from train_resnet18 import build_resnet18_cifar10, CIFAR10_MEAN, CIFAR10_STD
from fgsm_scratch   import fgsm_attack
from detector_pgd   import NormWrapper

WANDB_PROJECT  = "dlops-ass5-q2"
CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']
EPS = 0.03
N   = 10


def make_art_clf(model, device):
    wrapped   = NormWrapper(model, CIFAR10_MEAN, CIFAR10_STD, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    return PyTorchClassifier(
        model=wrapped, loss=criterion, optimizer=optimizer,
        input_shape=(3,32,32), nb_classes=10, clip_values=(0.,1.),
        device_type="gpu" if device.type=="cuda" else "cpu")


def to_01(x_norm: torch.Tensor) -> np.ndarray:
    """Reverse CIFAR-10 normalisation → [0,1] numpy NCHW."""
    m = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1,3,1,1)
    s = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1,3,1,1)
    return np.clip(x_norm.cpu().numpy() * s + m, 0, 1)


def np_to_wandb_image(x_np_chw, caption=""):
    """Convert NCHW or CHW float32 [0,1] to wandb.Image."""
    img = (x_np_chw.transpose(1,2,0) * 255).astype(np.uint8)
    return wandb.Image(img, caption=caption)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_ckpt", type=str,
                        default="../weights/Q2_resnet18_clean.pth")
    parser.add_argument("--data_dir",   type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_resnet18_cifar10().to(device)
    model.load_state_dict(torch.load(args.clean_ckpt, map_location=device))
    model.eval()

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set    = torchvision.datasets.CIFAR10(args.data_dir, False, True, transform=test_tf)
    loader      = DataLoader(test_set, N, False)
    imgs_norm, labs = next(iter(loader))
    imgs_norm       = imgs_norm.to(device)
    labs            = labs.to(device)

    imgs_01 = to_01(imgs_norm)          # shape (N,3,32,32) float32 [0,1]

    art_clf = make_art_clf(model, device)

    # Generate adversarial images
    criterion  = nn.CrossEntropyLoss()
    adv_scratch = to_01(fgsm_attack(model, imgs_norm, labs, EPS, criterion))

    adv_fgsm_art = FastGradientMethod(art_clf, eps=EPS).generate(imgs_01)
    adv_pgd      = ProjectedGradientDescentPyTorch(
                       art_clf, eps=EPS, eps_step=EPS/4, max_iter=20).generate(imgs_01)
    adv_bim      = BasicIterativeMethod(
                       art_clf, eps=EPS, eps_step=EPS/4, max_iter=20).generate(imgs_01)

    wandb.init(project=WANDB_PROJECT, name="attack_sample_visualizations")

    columns = ["Index", "Label",
               "Clean",
               "FGSM Scratch", "FGSM ART",
               "PGD", "BIM"]
    table = wandb.Table(columns=columns)

    for i in range(N):
        label = CIFAR10_LABELS[labs[i].item()]
        table.add_data(
            i, label,
            np_to_wandb_image(imgs_01[i],        f"Clean — {label}"),
            np_to_wandb_image(adv_scratch[i],    f"FGSM-scratch — {label}"),
            np_to_wandb_image(adv_fgsm_art[i],   f"FGSM-ART — {label}"),
            np_to_wandb_image(adv_pgd[i],        f"PGD — {label}"),
            np_to_wandb_image(adv_bim[i],        f"BIM — {label}"),
        )

    wandb.log({"attack_samples": table})

    # Also log flat image panels for each attack
    for name, adv in [("FGSM_Scratch", adv_scratch),
                      ("FGSM_ART",     adv_fgsm_art),
                      ("PGD",          adv_pgd),
                      ("BIM",          adv_bim)]:
        wandb.log({
            f"samples/{name}/clean":       [np_to_wandb_image(imgs_01[i])    for i in range(N)],
            f"samples/{name}/adversarial": [np_to_wandb_image(adv[i])        for i in range(N)],
        })

    wandb.finish()
    print("Uploaded 10-sample visualizations for all four attacks to WandB.")


if __name__ == "__main__":
    main()
