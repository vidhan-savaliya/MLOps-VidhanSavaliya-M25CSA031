"""
Utility functions and constants for the Goodreads genre classification project.
"""

import logging
import os
from typing import Dict

# Model configuration
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
DEVICE_NAME = 'cuda'  # Will fallback to 'cpu' if CUDA not available

# Training configuration
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 10
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Data configuration
SAMPLE_SIZE = 2000  # Reviews per genre
TRAIN_SIZE = 800    # Training samples per genre
TEST_SIZE = 200     # Test samples per genre
HEAD_SIZE = 10000   # Number of lines to read from each file

# Genre URLs for Goodreads dataset
GENRE_URL_DICT: Dict[str, str] = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

# Output paths
OUTPUT_DIR = './outputs'
MODEL_SAVE_DIR = './outputs/distilbert-reviews-genres'
RESULTS_DIR = './results'
LOGS_DIR = './logs'


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [OUTPUT_DIR, MODEL_SAVE_DIR, RESULTS_DIR, LOGS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logging.info(f"Ensured directories exist: {', '.join(directories)}")


def get_device() -> str:
    """
    Get the appropriate device for training/inference.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    import torch
    if torch.cuda.is_available() and DEVICE_NAME == 'cuda':
        device = 'cuda'
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logging.info("Using CPU device")
    return device
