"""
Data loading and preprocessing utilities for Goodreads genre classification.
"""

import gzip
import json
import logging
import random
from collections import defaultdict
import os
from typing import Dict, List, Tuple

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from .utils import GENRE_URL_DICT, SAMPLE_SIZE, TRAIN_SIZE, TEST_SIZE, HEAD_SIZE


class MyDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized text data.
    
    Args:
        encodings: Tokenized encodings from the tokenizer
        labels: List of integer labels
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_reviews(url: str, head: int = HEAD_SIZE, sample_size: int = SAMPLE_SIZE, data_dir: str = './data') -> List[str]:
    """
    Load and sample reviews from a gzipped JSON file URL.
    
    Args:
        url: URL to the gzipped JSON file
        head: Number of lines to read from the file
        sample_size: Number of reviews to randomly sample
        data_dir: Directory to cache downloaded files (Unused in streaming mode for speed)
        
    Returns:
        List of sampled review texts
    """
    logging.info(f"Loading reviews from {url}")
    
    # Download the file stream
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Read and parse the gzipped JSON from stream
    reviews = []
    # Use response.raw directly for streaming GZIP
    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= head:
                break
            try:
                review_data = json.loads(line)
                if 'review_text' in review_data:
                    reviews.append(review_data['review_text'])
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line {i}")
                continue
    
    # Sample reviews
    if len(reviews) > sample_size:
        reviews = random.sample(reviews, sample_size)
    
    logging.info(f"Loaded {len(reviews)} reviews")
    return reviews


def load_all_genres(genre_url_dict: Dict[str, str] = None) -> Dict[str, List[str]]:
    """
    Load reviews for all genres.
    
    Args:
        genre_url_dict: Dictionary mapping genre names to URLs
        
    Returns:
        Dictionary mapping genre names to lists of reviews
    """
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT
    
    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        try:
            reviews = load_reviews(url)
            genre_reviews_dict[genre] = reviews
            logging.info(f"Successfully loaded {len(reviews)} reviews for genre: {genre}")
        except Exception as e:
            logging.error(f"Failed to load reviews for genre {genre}: {str(e)}")
            raise
    
    return genre_reviews_dict


def create_train_test_split(
    genre_reviews_dict: Dict[str, List[str]],
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split reviews into training and test sets.
    
    Args:
        genre_reviews_dict: Dictionary mapping genres to review lists
        train_size: Number of training samples per genre
        test_size: Number of test samples per genre
        
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    
    for genre, reviews in genre_reviews_dict.items():
        # Ensure we have enough reviews
        required = train_size + test_size
        if len(reviews) < required:
            logging.warning(
                f"Genre {genre} has only {len(reviews)} reviews, "
                f"but {required} are required. Using all available."
            )
            train_size_actual = min(train_size, len(reviews) - test_size)
            test_size_actual = len(reviews) - train_size_actual
        else:
            train_size_actual = train_size
            test_size_actual = test_size
        
        # Split the data
        train_texts.extend(reviews[:train_size_actual])
        train_labels.extend([genre] * train_size_actual)
        
        test_texts.extend(reviews[train_size_actual:train_size_actual + test_size_actual])
        test_labels.extend([genre] * test_size_actual)
    
    logging.info(f"Created train set with {len(train_texts)} samples")
    logging.info(f"Created test set with {len(test_texts)} samples")
    
    return train_texts, train_labels, test_texts, test_labels


def create_label_mappings(labels: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Create bidirectional mappings between labels and IDs.
    
    Args:
        labels: List of label strings
        
    Returns:
        Tuple of (id2label, label2id) dictionaries
    """
    unique_labels = sorted(set(labels))
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    
    logging.info(f"Created label mappings for {len(unique_labels)} unique labels")
    logging.info(f"Labels: {unique_labels}")
    
    return id2label, label2id


def encode_labels(labels: List[str], label2id: Dict[str, int]) -> List[int]:
    """
    Convert string labels to integer IDs.
    
    Args:
        labels: List of string labels
        label2id: Mapping from labels to IDs
        
    Returns:
        List of integer label IDs
    """
    return [label2id[label] for label in labels]


def prepare_datasets(tokenizer, max_length: int = 512) -> Tuple[MyDataset, MyDataset, Dict[int, str], Dict[str, int]]:
    """
    Complete data preparation pipeline.
    
    Args:
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Tuple of (train_dataset, test_dataset, id2label, label2id)
    """
    logging.info("Starting data preparation pipeline")
    
    # Load all genre reviews
    genre_reviews_dict = load_all_genres()
    
    # Create train/test split
    train_texts, train_labels, test_texts, test_labels = create_train_test_split(genre_reviews_dict)
    
    # Create label mappings
    all_labels = train_labels + test_labels
    id2label, label2id = create_label_mappings(all_labels)
    
    # Encode labels
    train_labels_encoded = encode_labels(train_labels, label2id)
    test_labels_encoded = encode_labels(test_labels, label2id)
    
    # Tokenize texts
    logging.info("Tokenizing training texts...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    logging.info("Tokenizing test texts...")
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Create datasets
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    
    logging.info("Data preparation complete")
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset, id2label, label2id
