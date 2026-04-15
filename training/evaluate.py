"""Model evaluation utilities."""

import torch
from tqdm import tqdm
import numpy as np
from utils.metrics import calculate_metrics, get_classification_report

def evaluate(model, loader, device, verbose=True):
    """Evaluate model on dataset.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for evaluation data
        device: Device to evaluate on
        verbose: Print progress
    
    Returns:
        tuple: (metrics_dict, predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", disable=not verbose)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lex = batch["lex"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask, lex)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    if verbose:
        print("\nEvaluation Results:")
        for key, val in metrics.items():
            print(f"  {key.capitalize()}: {val:.4f}")
    
    return metrics, np.array(all_preds), np.array(all_labels)

def get_predictions(model, loader, device):
    """Get raw predictions from model.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device to run on
    
    Returns:
        tuple: (logits, predictions, labels)
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lex = batch["lex"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, lex)
            predictions = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.vstack(all_logits), np.array(all_preds), np.array(all_labels)