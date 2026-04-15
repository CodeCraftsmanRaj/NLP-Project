import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix

def ensure_results_dir(subdir=""):
    """Ensure results directory exists."""
    results_path = Path("results") / subdir
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

def plot_loss(losses, save_path="results/loss_curve.png", show=False):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, marker='o')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss plot saved: {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_cm(y_true, y_pred, labels=None, save_path="results/confusion_matrix.png", show=False):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    if labels:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=labels, yticklabels=labels, cbar=True)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_metrics(metrics_dict, save_path="results/metrics.png", show=False):
    """Plot metrics comparison (accuracy, precision, recall, F1)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics plot saved: {save_path}")
    if show:
        plt.show()
    plt.close()