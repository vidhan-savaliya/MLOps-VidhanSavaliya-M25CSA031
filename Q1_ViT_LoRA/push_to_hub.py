"""
push_to_hub.py - Push the best Q1 LoRA model weights to HuggingFace Hub.

Usage:
  python push_to_hub.py \
      --lora_dir ../weights/Q1_best_model \
      --repo_id Vidhansavaliya123/dlops-ass5-vit-lora \
      --rank 8 --alpha 8

Set HF_TOKEN env variable before running:
  export HF_TOKEN=your_huggingface_token
"""
import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, Repository, login

from model import build_lora_model
from dataset import get_dataloaders
from utils import evaluate
import torch.nn as nn


def push_lora_model(
    lora_dir: str,
    repo_id: str,
    rank: int,
    alpha: int,
    dropout: float,
):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN environment variable first.")
    login(token=token)

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload all files in the lora_dir
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")

    for f in lora_path.iterdir():
        if f.is_file():
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
            )
            print(f"  Uploaded: {f.name}")

    # Write a model card
    card = f"""---
language: en
license: mit
tags:
  - vision-transformer
  - lora
  - cifar-100
  - peft
  - image-classification
---

# ViT-S + LoRA fine-tuned on CIFAR-100

**Assignment:** DLOps Assignment 5 — Q1  
**Base model:** `timm/vit_small_patch16_224` (pretrained on ImageNet)  
**Fine-tuning method:** LoRA (PEFT) applied to attention QKV projections

## LoRA Configuration
| Parameter | Value |
|-----------|-------|
| Rank (r)  | {rank} |
| Alpha (α) | {alpha} |
| Dropout   | {dropout} |
| Target    | `attn.qkv` (fused Q/K/V) in all attention blocks |

## Performance
> Fill in test accuracy after evaluation.

## Usage
```python
from peft import PeftModel
import timm

base = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=100)
model = PeftModel.from_pretrained(base, "{repo_id}")
```
"""
    card_path = lora_path / "README.md"
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"\n✔ Model pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=str,
                        default="../weights/Q1_best_model")
    parser.add_argument("--repo_id",  type=str,
                        default="Vidhansavaliya123/dlops-ass5-vit-lora")
    parser.add_argument("--rank",     type=int,   default=8)
    parser.add_argument("--alpha",    type=int,   default=8)
    parser.add_argument("--dropout",  type=float, default=0.1)
    args = parser.parse_args()

    push_lora_model(
        lora_dir=args.lora_dir,
        repo_id=args.repo_id,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
