"""
Model initialization and configuration for DistilBERT sequence classification.
"""

import logging
from typing import Dict

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from .utils import MODEL_NAME


def get_tokenizer(model_name: str = MODEL_NAME) -> DistilBertTokenizerFast:
    """
    Load the DistilBERT tokenizer.
    
    Args:
        model_name: Name of the pre-trained model
        
    Returns:
        Loaded tokenizer
    """
    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    logging.info("Tokenizer loaded successfully")
    return tokenizer


def get_model(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    device: str
) -> DistilBertForSequenceClassification:
    """
    Load and configure the DistilBERT model for sequence classification.
    
    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of classification labels
        id2label: Mapping from label IDs to label names
        label2id: Mapping from label names to label IDs
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Configured model on the specified device
    """
    logging.info(f"Loading model: {model_name}")
    logging.info(f"Number of labels: {num_labels}")
    logging.info(f"Device: {device}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model loaded successfully")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_model_from_path(model_path: str, device: str) -> DistilBertForSequenceClassification:
    """
    Load a fine-tuned model from a local path.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    logging.info(f"Loading model from: {model_path}")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    logging.info("Model loaded successfully from path")
    return model


def load_model_from_hub(model_id: str, device: str) -> DistilBertForSequenceClassification:
    """
    Load a model from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model ID (username/model-name)
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    logging.info(f"Loading model from Hugging Face Hub: {model_id}")
    model = DistilBertForSequenceClassification.from_pretrained(model_id)
    model = model.to(device)
    logging.info("Model loaded successfully from Hub")
    return model
