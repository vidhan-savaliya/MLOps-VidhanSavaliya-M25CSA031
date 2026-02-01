# CIFAR-10 Classification with ResNet18

This repository contains a PyTorch implementation of ResNet18 for classifying CIFAR-10 images. It includes a custom dataloader, correct Train/Validation/Test splitting, FLOPs counting, and extensive visualizations logged to Weights & Biases (WandB).

## Project Structure

- `dataset.py`: Custom Dataset class with 45k/5k/10k Split and Augmentation.
- `model.py`: ResNet18 architecture adapted for 32x32 input size.
- `train.py`: Main training script with Validation loop and WandB logging.
- `utils.py`: Helper functions for FLOPs counting and Visualizations (Confusion Matrix, etc.).
- `report.tex`: LaTeX report template.
- `requirements.txt`: Dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model:

```bash
python train.py
```

This will:

- Download CIFAR-10 (if not present).
- Train for 30 epochs.
- Log metrics and graphs to WandB.

## Visualizations & Results

All training metrics and graphs are logged to WandB.

**Project Link**: [WandB Dashboard](https://wandb.ai/savaliya-indian-institute-of-technology-jodhpur/cifar10-lab-assignment/runs/r2qi5d4j)

The following visualizations are available in the **Media** or **Logs** section of the dashboard:

- **Prediction Examples**: `prediction_examples`
- **Gradient Flow**: `gradients`
- **Weight Distribution**: `weights`

## Author

Vidhan Savaliya
M25CSA031
Indian Institute of Technology Jodhpur

