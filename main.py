"""Main training and evaluation pipeline."""

import warnings

# Suppress transformers library deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import sys

from torch.utils.data import DataLoader

from config import BATCH_SIZE, DEVICE, XLEX_PATH
from data.dataset import FinancialDataset
from data.load_data import load_phrasebank
from models.model import EnhancedFinSentiBERT
from training.evaluate import evaluate
from training.train import save_checkpoint, train
from utils.metrics import print_metrics_summary, save_metrics_json
from utils.plots import ensure_results_dir, plot_cm, plot_loss, plot_metrics
from xlex.xlex import XLex


def main() -> None:
	"""Main execution function."""
	print("\n" + "=" * 60)
	print("Financial Sentiment Analysis Pipeline")
	print("=" * 60 + "\n")

	# Step 1: Load data
	print("[1/5] Loading dataset...")
	try:
		texts, labels = load_phrasebank()
		print(f"  ✓ Loaded {len(texts)} samples\n")
	except FileNotFoundError as e:
		print(f"  ✗ Error: {e}")
		sys.exit(1)

	# Step 2: Load XLex
	print("[2/5] Loading XLex...")
	if not Path(XLEX_PATH).exists():
		print(f"  ✗ XLex not found at {XLEX_PATH}")
		print("  Run 'python xlex/build_xlex.py' first.")
		sys.exit(1)

	xlex = XLex(XLEX_PATH)
	print(f"  ✓ XLex loaded ({XLEX_PATH})\n")

	# Step 3: Prepare data
	print("[3/5] Preparing data loaders...")
	dataset = FinancialDataset(texts, labels, xlex)
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	print(f"  ✓ DataLoader created (batch_size={BATCH_SIZE})\n")

	# Step 4: Train model
	print("[4/5] Training model...")
	model = EnhancedFinSentiBERT().to(DEVICE)
	print(f"  ✓ Model initialized (device={DEVICE})")

	losses, model = train(model, loader, DEVICE, verbose=True)
	print("  ✓ Training complete\n")

	# Step 5: Evaluate model
	print("[5/5] Evaluating model...")
	metrics, preds, y_true = evaluate(model, loader, DEVICE, verbose=True)
	print("  ✓ Evaluation complete\n")

	# Save results
	print("\nSaving results...")
	ensure_results_dir()

	plot_loss(losses, save_path="results/loss_curve.png", show=False)
	plot_cm(
		y_true,
		preds,
		labels=["Positive", "Negative", "Neutral"],
		save_path="results/confusion_matrix.png",
		show=False,
	)
	plot_metrics(metrics, save_path="results/metrics.png", show=False)
	save_metrics_json(metrics, save_path="results/metrics.json")

	print_metrics_summary(metrics, "Model")

	# Save model checkpoint
	save_checkpoint(
		model,
		None,
		0,
		losses[-1],
		save_path="models_checkpoint/best_model.pt",
	)

	print("\n" + "=" * 60)
	print("Training pipeline completed successfully!")
	print("Results saved to: results/")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()