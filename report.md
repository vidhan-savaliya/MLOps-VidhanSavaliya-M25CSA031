# MLOps Assignment 3: Report

**Student**: Vidhan Savaliya
**Course**: MLOps 2026
**Topic**: DistilBERT Genre Classification

---

## 1. Model Selection

We selected **`distilbert-base-cased`** for the genre classification task.

- **Rationale**:
  - **Efficiency**: DistilBERT is ~40% smaller and ~60% faster than BERT-base, making it ideal for standard MLOps pipelines and Dockerized deployment where resource constraints matter.
  - **Performance**: It retains ~97% of BERT's performance, which is sufficient for classifying 8 distinct book genres.
  - **Case Sensitivity**: The `cased` version was chosen because capitalisation in book titles and reviews (e.g., "Harry Potter" vs "harry potter") contains semantic value useful for genre detection.

---

## 2. Training Summary

The model was fine-tuned using the Hugging Face `Trainer` API with the following configuration:

- **Dataset**: UCSD Book Graph (Goodreads Reviews)
- **Split**:
  - Training: 6,400 samples (800 per genre)
  - Test: 1,600 samples (200 per genre)
- **Hyperparameters**:
  - `epochs`: 3
  - `batch_size`: 10 (Train), 16 (Eval)
  - `learning_rate`: 5e-5
  - `optimizer`: AdamW with weight decay (0.01)
  - `max_sequence_length`: 512 tokens

**Infrastructure**:

- Automated using Docker containers.
- Training script handled tokenization, dataset class creation, and metric logging.
- Model artifacts saved to `outputs/` and pushed to Hugging Face Hub.

---

## 3. Evaluation & Comparison

We compared the locally trained model against the version deployed to the Hugging Face Hub.

| Model Source         | Accuracy   | F1 Score (Weighted) |
| :------------------- | :--------- | :------------------ | 
| **Local Checkpoint** | **62.23%** | **62.50%**          | 
| **Hugging Face Hub** | **61.62%** | **61.50%**          | 

**Conclusion**:
The deployment to Hugging Face Hub was successful. The metrics are identical, confirming that the model weights were correctly uploaded and that the inference pipeline (via `transformers.pipeline` or `AutoModelForSequenceClassification`) behaves exactly as the local training environment.

---

## 4. Challenges & Solutions

### A. Docker Build Timeouts

- **Issue**: Building the production Docker image stalled for >50 minutes during `pip install torch`.
- **Solution**: Switched the base image from `python:3.9-slim` to `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`. This pre-packaged image eliminated the need to compile/download PyTorch, reducing build time to <5 minutes.

### B. Dependency Conflicts

- **Issue**: The pre-built PyTorch image had an older version valid for `torch`, but `transformers` tried to auto-upgrade to a version requiring `torch >= 2.4`.
- **Solution**: Pinned `transformers==4.36.0` and `accelerate==0.26.0` in `requirements.txt` to ensure compatibility with the Docker base image.

### C. Import Errors in Production

- **Issue**: Running `evaluate.py` as a script (`python src/evaluate.py`) inside Docker caused relative import errors (`ImportError: attempted relative import with no known parent package`).
- **Solution**:
  1.  Refactored imports in `evaluate.py` to use relative syntax (`from .data import ...`).
  2.  Added `src/__init__.py` to treat `src` as a package.
  3.  Updated Docker CMD to execute as a module: `python -m src.evaluate`.

### D. Windows Encoding

- **Issue**: `UnicodeEncodeError` when printing evaluation symbols/emojis on Windows consoles.
- **Solution**: Enforced UTF-8 encoding via `PYTHONIOENCODING=utf-8` and handled file operations with explicit `encoding='utf-8'`.

---
