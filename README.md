# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Roll No:** M25CSA031  
**Task:** English → Hindi Neural Machine Translation  
**Model:** PyTorch Transformer  
**Dataset:** English-Hindi parallel corpus

---

## Repository Structure

```
├── M25CSA031_ass_4_tuned_en_to_hi.ipynb   # Main notebook (baseline + Ray Tune)
├── M25CSA031_ass_4_best_model.pth          # Saved weights of best model
├── M25CSA031_ass_4_report.pdf              # 2-page assignment report
├── M25CSA031_best_config.json              # Best hyperparameter config (JSON)
├── English-Hindi.tsv                       # Dataset (download separately)
└── README.md
```

---

## Results Summary

| Metric | Baseline (100 epochs) | Best Model (50 epochs) |
|---|---|---|
| Training Time | 42 min 29.8 sec | ~21 min |
| Final Loss | 0.0492 | 0.0798 |
| BLEU Score | 18.32 | **18.48**  |
| Epochs | 100 | **50** |

> Goal achieved: matched and exceeded baseline BLEU in **50 fewer epochs**.

---

## Best Hyperparameter Configuration

Found by Ray Tune + OptunaSearch across 20 trials:

| Hyperparameter | Value |
|---|---|
| Learning Rate | 0.000279 |
| Batch Size | 64 |
| Num Attention Heads | 8 |
| Feed-Forward Dim | 1024 |
| Dropout | 0.101 |

---

## How to Run

### 1. Open in Google Colab
Upload `M25CSA031_ass_4_tuned_en_to_hi.ipynb` to [colab.research.google.com](https://colab.research.google.com)  
Enable GPU: **Runtime → Change runtime type → T4 GPU**

### 2. Upload Dataset
Upload `English-Hindi.tsv` to the Colab session files panel.

### 3. Run All Cells
The notebook is structured in 3 parts:

- **Part 1** — Baseline training (100 epochs, hardcoded hyperparams)
- **Part 2** — Ray Tune + Optuna sweep (20 trials, max 30 epochs each)
- **Part 3** — Retrain best config + BLEU evaluation

### 4. Install Dependencies
The notebook handles this automatically:
```bash
pip install ray[tune] optuna
```

---

## Hyperparameter Search Details

- **Search Algorithm:** OptunaSearch (TPE sampler)
- **Scheduler:** ASHAScheduler (grace_period=5, reduction_factor=2)
- **Trials:** 20 total, max 30 epochs each
- **Search Space:** 5 hyperparameters (lr, batch_size, num_heads, d_ff, dropout)

---

## Requirements

```
torch
ray[tune]
optuna
pandas
matplotlib
seaborn
nltk
tqdm
```

---

## Dataset

Download the dataset from the assignment Google Drive link and place `English-Hindi.tsv` in the same directory as the notebook.
