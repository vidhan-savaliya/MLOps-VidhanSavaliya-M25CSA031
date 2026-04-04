# -*- coding: utf-8 -*-
"""
push_all_to_hf.py
=================
Pushes ALL required model weights to HuggingFace Hub:
  - Q1: Best LoRA model (r=4, alpha=4)
  - Q2: ResNet-18 clean weights

Usage:
    python push_all_to_hf.py --hf_token YOUR_HF_TOKEN

Requirements: pip install huggingface_hub
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


# ─── Config ───────────────────────────────────────────────────────────────────
HF_USERNAME = "Vidhansavaliya123"

Q1_CARD = (
    "---\n"
    "language: en\n"
    "license: mit\n"
    "tags:\n"
    "  - vision-transformer\n"
    "  - lora\n"
    "  - peft\n"
    "  - cifar-100\n"
    "  - image-classification\n"
    "  - dlops\n"
    "---\n\n"
    "# ViT-Small + LoRA - Best Model (CIFAR-100)\n\n"
    "**Assignment:** DLOps Assignment 5 - Q1\n"
    "**Author:** Vidhan Savaliya (M25CSA031, IIT Jodhpur)\n"
    "**Base model:** `timm/vit_small_patch16_224` pre-trained on ImageNet\n"
    "**Fine-tuning:** LoRA (PEFT) on fused Q/K/V attention projections\n\n"
    "## LoRA Configuration (Best / Optuna-selected)\n"
    "| Parameter | Value |\n"
    "|-----------|-------|\n"
    "| Rank (r)  | 4     |\n"
    "| Alpha     | 4     |\n"
    "| Dropout   | 0.1   |\n"
    "| Target modules | `attn.qkv` (all attention blocks) |\n"
    "| Trainable params | 112,228 / 21,816,392 (0.51%) |\n\n"
    "## Performance on CIFAR-100\n"
    "| Split | Accuracy |\n"
    "|-------|----------|\n"
    "| Train (epoch 9) | 93.70% |\n"
    "| Validation (best) | **90.88%** |\n"
    "| Test | **90.88%** |\n\n"
    "Baseline (head-only): 78.12% -> **+12.76% improvement with LoRA**\n\n"
    "## WandB\n"
    "https://wandb.ai/vidhan-savaliya-indian-institute-of-technology-jodhpur/dlops-ass5-q1/overview\n\n"
    "## Usage\n"
    "```python\n"
    "import timm\n"
    "from peft import PeftModel\n\n"
    "base = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=100)\n"
    "model = PeftModel.from_pretrained(base, 'Vidhansavaliya123/dlops-ass5-q1-vit-lora-best')\n"
    "model.eval()\n"
    "```\n"
)

Q2_CARD = (
    "---\n"
    "language: en\n"
    "license: mit\n"
    "tags:\n"
    "  - resnet\n"
    "  - cifar-10\n"
    "  - image-classification\n"
    "  - adversarial-robustness\n"
    "  - dlops\n"
    "---\n\n"
    "# ResNet-18 - Trained from Scratch on CIFAR-10\n\n"
    "**Assignment:** DLOps Assignment 5 - Q2\n"
    "**Author:** Vidhan Savaliya (M25CSA031, IIT Jodhpur)\n"
    "**Architecture:** ResNet-18 (no pre-training)\n"
    "**Dataset:** CIFAR-10\n\n"
    "## Training Details\n"
    "| Parameter | Value |\n"
    "|-----------|-------|\n"
    "| Epochs    | 50    |\n"
    "| Optimizer | SGD + Momentum |\n"
    "| LR Schedule | Cosine Annealing |\n"
    "| Batch Size | 128  |\n\n"
    "## Performance\n"
    "| Metric | Value |\n"
    "|--------|-------|\n"
    "| Final Train Accuracy | 99.78% |\n"
    "| Final Val Accuracy   | 92.82% |\n"
    "| **Final Test Accuracy** | **93.79%** |\n"
    "| Assignment Target    | >= 72% PASSED |\n\n"
    "## Adversarial Robustness (FGSM)\n"
    "| Epsilon | Test Accuracy |\n"
    "|---------|---------------|\n"
    "| 0.00 | 93.79% |\n"
    "| 0.01 | 75.28% |\n"
    "| 0.03 | 49.34% |\n"
    "| 0.10 | 32.10% |\n"
    "| 0.30 | 21.13% |\n\n"
    "## WandB\n"
    "https://wandb.ai/vidhan-savaliya-indian-institute-of-technology-jodhpur/dlops-ass5-q2?nw=nwuservidhansavaliya\n\n"
    "## Usage\n"
    "```python\n"
    "import torch\n"
    "import torchvision.models as models\n\n"
    "model = models.resnet18(pretrained=False, num_classes=10)\n"
    "state_dict = torch.load('Q2_resnet18_clean.pth', map_location='cpu')\n"
    "model.load_state_dict(state_dict)\n"
    "model.eval()\n"
    "```\n"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def push_folder(api, local_dir, repo_id, card):
    """Upload all files from a directory + write a README/model card."""
    print("\n  Uploading folder: %s -> %s" % (local_dir, repo_id))
    card_file = local_dir / "README.md"
    card_file.write_text(card, encoding="utf-8")
    for f in sorted(local_dir.iterdir()):
        if f.is_file():
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
            )
            print("    OK %s  (%.1f KB)" % (f.name, f.stat().st_size / 1024))
    print("  -> https://huggingface.co/%s" % repo_id)


def push_single_file(api, local_file, repo_id, card):
    """Upload a single weights file + a README model card."""
    print("\n  Uploading file: %s -> %s" % (local_file.name, repo_id))
    # Write card to a temp location
    tmp_card = local_file.parent / "_README_tmp.md"
    tmp_card.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(tmp_card),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    tmp_card.unlink(missing_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=local_file.name,
        repo_id=repo_id,
    )
    size_mb = local_file.stat().st_size / (1024 * 1024)
    print("    OK %s  (%.1f MB)" % (local_file.name, size_mb))
    print("  -> https://huggingface.co/%s" % repo_id)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: Provide --hf_token or set HF_TOKEN env var.")
        sys.exit(1)

    login(token=args.hf_token, add_to_git_credential=False)
    api = HfApi()

    base_dir = Path(__file__).parent
    sep = "=" * 60

    # ── Q1: Best LoRA model (r=4, alpha=4) ────────────────────────────────────
    print("\n" + sep)
    print("Q1: Pushing Best LoRA Model (r=4, alpha=4, dropout=0.1)")
    print(sep)
    q1_repo = "%s/dlops-ass5-q1-vit-lora-best" % HF_USERNAME
    q1_local = base_dir / "weights" / "Q1_lora_r4_a4_best"
    api.create_repo(repo_id=q1_repo, repo_type="model", exist_ok=True)
    push_folder(api, q1_local, q1_repo, Q1_CARD)

    # ── Q2: ResNet-18 clean weights ────────────────────────────────────────────
    print("\n" + sep)
    print("Q2: Pushing ResNet-18 Clean Weights (93.79% CIFAR-10)")
    print(sep)
    q2_repo  = "%s/dlops-ass5-q2-resnet18-cifar10" % HF_USERNAME
    q2_local = base_dir / "weights" / "Q2_resnet18_clean.pth"
    api.create_repo(repo_id=q2_repo, repo_type="model", exist_ok=True)
    push_single_file(api, q2_local, q2_repo, Q2_CARD)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + sep)
    print("ALL MODELS PUSHED TO HUGGINGFACE SUCCESSFULLY")
    print(sep)
    print("  Q1 Best LoRA : https://huggingface.co/%s" % q1_repo)
    print("  Q2 ResNet-18 : https://huggingface.co/%s" % q2_repo)
    print()


if __name__ == "__main__":
    main()
