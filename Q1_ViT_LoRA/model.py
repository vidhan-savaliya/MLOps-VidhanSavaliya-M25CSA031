"""
model.py - ViT-S model builder for baseline and LoRA fine-tuning
"""
import timm
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_baseline_model(num_classes: int = 100) -> nn.Module:
    """
    ViT-S pretrained on ImageNet.
    Backbone fully frozen; only classification head is trainable.
    """
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=True,
        num_classes=num_classes,
    )
    for name, param in model.named_parameters():
        param.requires_grad = ('head' in name)

    trainable = count_trainable_params(model)
    total     = count_total_params(model)
    print(f"[Baseline] Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")
    return model


def build_lora_model(
    num_classes: int = 100,
    rank: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    ViT-S with LoRA adapters injected into all QKV attention projections.
    Classification head remains trainable via modules_to_save.
    In timm ViT-S, Q/K/V share a single fused linear layer named 'qkv'.
    """
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=True,
        num_classes=num_classes,
    )

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv"],        # fused Q/K/V in every attention block
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["head"],      # keep new classification head trainable
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
