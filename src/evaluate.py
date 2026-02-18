"""
Evaluation script for DistilBERT genre classification model.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer, TrainingArguments

# Add src to path if running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .data import prepare_datasets
from .model import get_tokenizer, load_model_from_path, load_model_from_hub
from .utils import (
    setup_logging,
    ensure_directories,
    get_device,
    MODEL_NAME,
    MAX_LENGTH,
    PER_DEVICE_EVAL_BATCH_SIZE,
    OUTPUT_DIR
)


def compute_metrics(pred):
    """
    Compute accuracy metric for evaluation.
    
    Args:
        pred: Prediction output from the model
        
    Returns:
        Dictionary with accuracy metric
    """
    from sklearn.metrics import accuracy_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DistilBERT genre classification model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to local model directory'
    )
    parser.add_argument(
        '--model-from-hub',
        type=str,
        default=None,
        help='Model ID from Hugging Face Hub (username/model-name)'
    )
    parser.add_argument(
        '--tokenizer-name',
        type=str,
        default=MODEL_NAME,
        help='Tokenizer name (default: distilbert-base-cased)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_LENGTH,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=PER_DEVICE_EVAL_BATCH_SIZE,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=os.path.join(OUTPUT_DIR, 'evaluation_results.json'),
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting evaluation script")
    logging.info(f"Arguments: {vars(args)}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Get device
    device = get_device()
    
    # Validate arguments
    if args.model_path is None and args.model_from_hub is None:
        logging.error("Either --model-path or --model-from-hub must be specified")
        sys.exit(1)
    
    if args.model_path is not None and args.model_from_hub is not None:
        logging.error("Only one of --model-path or --model-from-hub should be specified")
        sys.exit(1)
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = get_tokenizer(args.tokenizer_name)
    
    # Prepare test dataset
    logging.info("Preparing test dataset...")
    _, test_dataset, id2label, label2id = prepare_datasets(
        tokenizer,
        max_length=args.max_length
    )
    
    # Load model
    if args.model_path:
        logging.info(f"Loading model from local path: {args.model_path}")
        model = load_model_from_path(args.model_path, device)
        model_source = f"local:{args.model_path}"
    else:
        logging.info(f"Loading model from Hugging Face Hub: {args.model_from_hub}")
        model = load_model_from_hub(args.model_from_hub, device)
        model_source = f"hub:{args.model_from_hub}"
    
    # Configure evaluation arguments
    eval_args = TrainingArguments(
        output_dir='./eval_results',
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
    )
    
    # Initialize Trainer for evaluation
    logging.info("Initializing Trainer for evaluation...")
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Run evaluation
    logging.info("Running evaluation...")
    logging.info("=" * 80)
    eval_results = trainer.evaluate()
    
    # Get predictions
    logging.info("Generating predictions...")
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions.argmax(-1)
    true_labels = predictions_output.label_ids
    
    # Generate classification report
    logging.info("Generating classification report...")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        output_dict=True
    )
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Prepare results
    results = {
        'model_source': model_source,
        'evaluation_metrics': eval_results,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'label_mapping': {
            'id2label': id2label,
            'label2id': label2id
        }
    }
    
    # Log results
    logging.info("=" * 80)
    logging.info("Evaluation Results:")
    logging.info(f"Model source: {model_source}")
    logging.info(f"Test samples: {len(test_dataset)}")
    logging.info(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    logging.info(f"Loss: {eval_results['eval_loss']:.4f}")
    
    logging.info("\nPer-class metrics:")
    for label_name in target_names:
        metrics = class_report[label_name]
        logging.info(
            f"  {label_name:25s} - "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1-score']:.4f}"
        )
    
    logging.info(f"\nMacro avg F1: {class_report['macro avg']['f1-score']:.4f}")
    logging.info(f"Weighted avg F1: {class_report['weighted avg']['f1-score']:.4f}")
    
    # Save results to file
    logging.info(f"\nSaving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved successfully")
    
    # Print classification report in text format
    logging.info("\nDetailed Classification Report:")
    logging.info("\n" + classification_report(
        true_labels,
        predictions,
        target_names=target_names
    ))
    
    logging.info("=" * 80)
    logging.info("Evaluation script completed successfully!")


if __name__ == '__main__':
    main()
