"""
Training script for DistilBERT genre classification model.
"""

import argparse
import logging
import os
import sys

import torch
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

# Add src to path if running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import prepare_datasets
from model import get_tokenizer, get_model
from utils import (
    setup_logging,
    ensure_directories,
    get_device,
    MODEL_NAME,
    MAX_LENGTH,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MODEL_SAVE_DIR,
    RESULTS_DIR,
    LOGS_DIR
)


def compute_metrics(pred):
    """
    Compute accuracy metric for evaluation.
    
    Args:
        pred: Prediction output from the model
        
    Returns:
        Dictionary with accuracy metric
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DistilBERT for genre classification')
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help='Pre-trained model name'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_LENGTH,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=PER_DEVICE_TRAIN_BATCH_SIZE,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=PER_DEVICE_EVAL_BATCH_SIZE,
        help='Evaluation batch size per device'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=MODEL_SAVE_DIR,
        help='Directory to save the model'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push model to Hugging Face Hub after training'
    )
    parser.add_argument(
        '--hub-model-id',
        type=str,
        default=None,
        help='Model ID for Hugging Face Hub (username/model-name)'
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
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting training script")
    logging.info(f"Arguments: {vars(args)}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Get device
    device = get_device()
    
    # Disable wandb logging
    os.environ['WANDB_DISABLED'] = 'true'
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = get_tokenizer(args.model_name)
    
    # Prepare datasets
    logging.info("Preparing datasets...")
    train_dataset, test_dataset, id2label, label2id = prepare_datasets(
        tokenizer,
        max_length=args.max_length
    )
    
    # Initialize model
    logging.info("Initializing model...")
    model = get_model(
        model_name=args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        device=device
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to=[],  # Disable wandb
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )
    
    # Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logging.info("Starting training...")
    logging.info("=" * 80)
    train_result = trainer.train()
    
    # Log training results
    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info(f"Training loss: {train_result.training_loss:.4f}")
    logging.info(f"Training steps: {train_result.global_step}")
    
    # Save the model
    logging.info(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Model saved successfully")
    
    # Evaluate on test set
    logging.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logging.info("Evaluation results:")
    for key, value in eval_results.items():
        logging.info(f"  {key}: {value:.4f}")
    
    # Push to Hub if requested
    if args.push_to_hub:
        if args.hub_model_id is None:
            logging.error("--hub-model-id must be specified when using --push-to-hub")
            sys.exit(1)
        
        logging.info(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
        try:
            trainer.push_to_hub(commit_message="Training complete")
            logging.info("Model successfully pushed to Hub")
        except Exception as e:
            logging.error(f"Failed to push to Hub: {str(e)}")
            logging.error("Make sure you are logged in with: huggingface-cli login")
            sys.exit(1)
    
    logging.info("Training script completed successfully!")


if __name__ == '__main__':
    main()
