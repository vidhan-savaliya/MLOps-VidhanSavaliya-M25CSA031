import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
import wandb
import io
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def count_flops(model, input_size=(1, 3, 32, 32)):
    """
    Counts FLOPs for a given model.
    """
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = 2 * macs # Approximate FLOPs as 2x MACs
    print(f"Model Params: {params/1e6:.2f}M")
    print(f"Model FLOPs: {flops/1e6:.2f}M")
    return flops, params

def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers.
    Returns a PIL Image object to be logged to wandb.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            # Check if gradient is None (detached or unused parameter)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
            else:
                ave_grads.append(0)
                max_grads.append(0)
                
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # Zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_weight_distribution(named_parameters):
    """
    Plots the distribution of weights.
    Returns a PIL Image object.
    """
    weights = []
    for n, p in named_parameters:
        if "weight" in n and p.requires_grad:
            weights.extend(p.view(-1).detach().cpu().numpy())
            
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, alpha=0.7)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots the confusion matrix.
    Returns a PIL Image object.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_class_accuracy(y_true, y_pred, classes):
    """
    Plots the accuracy for each class.
    Returns a PIL Image object.
    """
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    c = (np.array(y_pred) == np.array(y_true)).squeeze()
    for i in range(len(y_true)):
        label = y_true[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
        
    accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(classes))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, accuracies, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 105)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_predictions(model, device, loader, classes, num_images=10):
    """
    Plots a batch of test images with predicted and actual labels.
    Returns a PIL Image object.
    """
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()
    
    # Denormalize
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    fig = plt.figure(figsize=(15, 6))
    for idx in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        img = images[idx].clone()
        # Un-normalize
        for t, m, s in zip(img, stats[0], stats[1]):
            t.mul_(s).add_(m)
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        ax.set_title(f"Pred: {classes[preds[idx]]}\\nTrue: {classes[labels[idx]]}",
                     color=("green" if preds[idx]==labels[idx] else "red"))
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img
