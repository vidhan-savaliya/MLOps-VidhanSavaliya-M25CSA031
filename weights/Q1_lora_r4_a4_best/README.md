---
language: en
license: mit
tags:
  - vision-transformer
  - lora
  - peft
  - cifar-100
  - image-classification
  - dlops
---

# ViT-Small + LoRA - Best Model (CIFAR-100)

**Assignment:** DLOps Assignment 5 - Q1
**Author:** Vidhan Savaliya (M25CSA031, IIT Jodhpur)
**Base model:** `timm/vit_small_patch16_224` pre-trained on ImageNet
**Fine-tuning:** LoRA (PEFT) on fused Q/K/V attention projections

## LoRA Configuration (Best / Optuna-selected)
| Parameter | Value |
|-----------|-------|
| Rank (r)  | 4     |
| Alpha     | 4     |
| Dropout   | 0.1   |
| Target modules | `attn.qkv` (all attention blocks) |
| Trainable params | 112,228 / 21,816,392 (0.51%) |

## Performance on CIFAR-100
| Split | Accuracy |
|-------|----------|
| Train (epoch 9) | 93.70% |
| Validation (best) | **90.88%** |
| Test | **90.88%** |

Baseline (head-only): 78.12% -> **+12.76% improvement with LoRA**

## WandB
https://wandb.ai/vidhan-savaliya-indian-institute-of-technology-jodhpur/dlops-ass5-q1/overview

## Usage
```python
import timm
from peft import PeftModel

base = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=100)
model = PeftModel.from_pretrained(base, 'Vidhansavaliya123/dlops-ass5-q1-vit-lora-best')
model.eval()
```
