# DistilBERT Genre Classification

> **"Don't judge a book by its cover... let AI judge it by its review!"**

Welcome to the **Goodreads Genre Classifier**! This project uses a fine-tuned **DistilBERT** model to automatically categorize book reviews into 8 different genres. Whether it's a spooky mystery or a heartwarming romance, this model knows what's up.

---

## Quick Links

- **Hugging Face Model**: [Vidhansavaliya123/distilbert-goodreads-genres](https://huggingface.co/Vidhansavaliya123/distilbert-goodreads-genres)
- **Docker Image**: `goodreads-genre-eval-prod` (Build instructions below)

---

## Performance

We trained this model on the UCSD Book Graph dataset (6,400 reviews) and achieved the following results on the test set:

| Metric       | Score      |
| :----------- | :--------- |
| **Accuracy** | **~61.6%** |
| **F1 Score** | **~61.5%** |

_(Not bad for a distilled model handling 8 complex genres!)_

### The Genres

The model can predict:
`Poetry`, `Children`, `Comics & Graphic`, `Fantasy & Paranormal`, `History & Biography`, `Mystery, Thriller & Crime`, `Romance`, `Young Adult`

---

## How to Run It (The Easy Way)

You don't need to install Python or mess with dependencies. We've Dockerized everything for you!

### 1. Evaluate the Model (Production Ready)

This command will automatically pull the model from Hugging Face and run the evaluation script on any machine.

```bash
# Build the image
docker build -t goodreads-genre-eval-prod -f Dockerfile.production .

# Run it!
docker run --rm goodreads-genre-eval-prod
```

### 2. Check the Code

If you prefer running things locally:

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run evaluation**:
    ```bash
    python -m src.evaluate --model-from-hub Vidhansavaliya123/distilbert-goodreads-genres
    ```

---

## Model Architecture

We chose **DistilBERT** (`distilbert-base-cased`) because it's the perfect balance of speed and smarts:

- **40% smaller** than BERT.
- **60% faster** inference.
- **97% of the performance** of the original BERT.

It's trained to handle case-sensitive text, which is great for book titles and proper nouns found in reviews.

---

## Project Structure

Here's what you'll find in the repo:

- `src/`: The brains of the operation (Training, Evaluation, Data Loading).
- `Dockerfile.production`: The magic file that makes deployment easy.
- `requirements.txt`: The list of ingredients (libraries).
- `evaluation_results.json`: Detailed report cards for the model.

---

## Credits

- **Dataset**: UCSD Book Graph (Goodreads Reviews)
- **Library**: Hugging Face Transformers defaults
- **Tagline**: Written by a human (me), optimized by AI.

---

_For questions, feel free to reach out or open an issue! Happy reading!_
