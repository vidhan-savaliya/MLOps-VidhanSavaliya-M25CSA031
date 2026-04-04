"""
dataset.py - CIFAR-100 data loading for ViT-S fine-tuning
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

# ViT-S was pretrained on ImageNet — use ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224  # ViT-S expects 224x224


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomCrop(IMAGE_SIZE, padding=IMAGE_SIZE // 8),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-100."""

    # Two instances so train/val have independent transforms
    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=get_transforms(True))
    full_val_src = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=get_transforms(False))
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=get_transforms(False))

    n = len(full_train)
    val_n = int(n * val_fraction)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=gen).tolist()

    train_set = Subset(full_train,   idx[val_n:])
    val_set   = Subset(full_val_src, idx[:val_n])

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, **kw)

    print(f"[Dataset] Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle',
    'bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
    'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach',
    'couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish',
    'forest','fox','girl','hamster','house','kangaroo','keyboard','lamp',
    'lawn_mower','leopard','lion','lizard','lobster','man','maple_tree',
    'motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
    'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate',
    'poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake',
    'spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank',
    'telephone','television','tiger','tractor','train','trout','tulip','turtle',
    'wardrobe','whale','willow_tree','wolf','woman','worm',
]
