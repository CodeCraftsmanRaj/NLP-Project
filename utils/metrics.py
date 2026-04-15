"""Metrics calculation and reporting utilities."""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json
from pathlib import Path

def calculate_metrics(y_true, y_pred, labels=None):
    """Calculate all metrics for classification task.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names for reporting
    
    Returns:
        dict: Dictionary with all metrics
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return metrics

def get_classification_report(y_true, y_pred, labels=None):
    """Get detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names
    
    Returns:
        str: Formatted classification report
    """
    return classification_report(y_true, y_pred, target_names=labels)

def save_metrics_json(metrics, save_path="results/metrics.json"):
    """Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {save_path}")

def print_metrics_summary(metrics, name="Model"):
    """Print formatted metrics summary.
    
    Args:
        metrics: Dictionary of metrics
        name: Name to display
    """
    print(f"\n{'='*50}")
    print(f"{name} Performance Summary")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"{key.capitalize():15s}: {value:.4f}")
    print(f"{'='*50}\n")