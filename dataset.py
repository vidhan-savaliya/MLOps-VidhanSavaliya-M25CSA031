import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np

class CustomCIFAR10Dataset(Dataset):
    """
    Custom Dataset for CIFAR-10.
    Wraps torchvision.datasets.CIFAR10 but allows for custom transformations and logic.
    """
    def __init__(self, root, train=True, download=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Load the underlying CIFAR-10 dataset
        self.cifar_data = datasets.CIFAR10(root=root, train=train, download=download)
        self.data = self.cifar_data.data
        self.targets = self.cifar_data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[idx], self.targets[idx]

        # Convert numpy array to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_dataloaders(batch_size=128, num_workers=2, root='./data', valid_size=0.1):
    """
    Returns train, validation, and test dataloaders.
    """
    # Define transforms
    # Standard CIFAR-10 normalization and augmentation
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Initialize Datasets
    # We need the full training set first, but we need to split it.
    # To apply different transforms to train and val, we can create two instances of the dataset.
    full_train_dataset = datasets.CIFAR10(root=root, train=True, download=True)
    
    # Create split indices
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    # Shuffle indices ensures random split, but fixing seed is better for reproducibility if needed.
    # Here we just use a consistent random split if we wanted, but np.random.shuffle is fine.
    # Ideally, we shuffle.
    np.random.seed(42) # For reproducibility
    np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # Custom Dataset Wrapper to apply specific transforms
    class SubsetWrapper(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                # subset[index] returns (img, target) where img is PIL Image
                # We need to make sure we don't double transform if the underlying dataset already did?
                # The underlying CIFAR10 (full_train_dataset) has transform=None by default above.
                pass
            # Applying transform to PIL image
            return self.transform(x), y
        
        def __len__(self):
            return len(self.subset)

    # Actually, the CustomCIFAR10Dataset in existing code wraps CIFAR10. 
    # Let's reuse the logic but be careful.
    # The existing CustomCIFAR10Dataset takes 'train' arg and loads CIFAR10 internally.
    # To avoid re-downloading/loading multiple times, we can modify it or just accept the split.
    # Simplest way: separate indices and use Subset, but Subset doesn't allow easy transform overriding 
    # unless the underlying dataset doesn't have transforms.
    
    # Let's instantiate the underlying data once (or just let the wrapper do it, it's cached).
    # We will use the CustomCIFAR10Dataset but we need to pass indices to it? 
    # No, CustomCIFAR10Dataset loads ALL data.
    
    # Better approach:
    # 1. Load full train data points (train=True)
    # 2. Split indices.
    # 3. Create two Subsets.
    # 4. Wrap Subsets to apply transforms?
    # Or strict to the CustomCIFAR10Dataset which inherits Dataset.
    
    train_dataset_base = CustomCIFAR10Dataset(root=root, train=True, download=True, transform=None)
    # Access internal data to split
    num_train = len(train_dataset_base)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # We need a way to apply different transforms. 
    # Let's modify CustomCIFAR10Dataset to accept indices? Or just use Subset and a wrapper.
    
    # Let's define a Clean Wrapper here or use the one above.
    # Actually, CustomCIFAR10Dataset implementation in this file (lines 7-41)
    # It takes `transform` in __init__.
    # So we can create TWO instances of CustomCIFAR10Dataset.
    # One with train_transform, one with test_transform.
    # Then use Subset on them with the respective indices?
    # Yes, that works. The data is the same, just the indices differ.
    
    train_set_full = CustomCIFAR10Dataset(root=root, train=True, download=True, transform=train_transform)
    valid_set_full = CustomCIFAR10Dataset(root=root, train=True, download=True, transform=test_transform) # Note: train=True for valid split source
    
    train_dataset = torch.utils.data.Subset(train_set_full, train_idx)
    valid_dataset = torch.utils.data.Subset(valid_set_full, valid_idx)
    
    test_dataset = CustomCIFAR10Dataset(root=root, train=False, download=True, transform=test_transform)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader
