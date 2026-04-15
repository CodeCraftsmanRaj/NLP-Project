# Financial Sentiment Analysis with XLex + EnhancedFinSentiBERT

## Overview

This project implements a financial sentiment analysis system based on:

* **EnhancedFinSentiBERT architecture** (multi-branch transformer model)
* **XLex (Explainable Lexicon)** generated using SHAP
* **Comprehensive training and inference pipelines**

The model combines:

1. **BERT semantic understanding**
2. **Explainable lexicon features (XLex)**
3. **Neutral sentiment feature extraction**

The goal is to improve sentiment classification in financial text with **interpretability** and **explainability**.

---

## Features

* **Multi-branch neural architecture:**
  * BERT branch (semantic features)
  * Lexicon branch (XLex features)
  * Neutral feature branch
  
* **Explainable lexicon generation** using SHAP
* **Comprehensive training pipeline** with progress tracking
* **Advanced evaluation metrics** (accuracy, precision, recall, F1)
* **Rich visualizations:**
  * Training loss curves
  * Confusion matrices
  * Metrics comparison charts
* **Production-ready inference** with confidence scores
* **Model checkpointing** and resumable training

---

## Project Structure

```
nlp-project/
│
├── config.py                 # Global configuration
├── main.py                   # Main training pipeline
├── inference.py              # Inference and prediction module
│
├── data/
│   ├── load_data.py         # Data loading utilities
│   ├── dataset.py           # PyTorch Dataset class
│   └── __pycache__/
│
├── models/
│   └── model.py             # EnhancedFinSentiBERT model
│
├── xlex/
│   ├── build_xlex.py        # XLex generation from SHAP
│   └── xlex.py              # XLex encoding class
│
├── training/
│   ├── train.py             # Training functions with checkpointing
│   └── evaluate.py          # Evaluation and metrics
│
├── utils/
│   ├── tokenizer.py         # BERT tokenization
│   ├── plots.py             # Visualization functions
│   └── metrics.py           # Metrics calculation and reporting
│
├── results/                 # Generated plots and metrics (auto-created)
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── metrics.png
│   └── metrics.json
│
├── models_checkpoint/       # Model checkpoints (auto-created)
│   └── best_model.pt
│
├── logs/                    # Training logs (auto-created)
│
├── FinancialPhraseBank-v1.0/  # Dataset directory
│
├── README.md                # This file
├── pyproject.toml          # Project dependencies
└── .gitignore              # Git ignore rules
```

---

## Dataset

This project uses the **Financial PhraseBank** dataset.

### Required File
```
FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
```

### Format
```
sentence@label
```
Where label ∈ {positive, negative, neutral}

---

## Installation

### 1. Create Environment

Using `uv` (recommended):
```bash
uv venv venv
source venv/bin/activate
```

Or using `pip`:
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
# or
pip install torch transformers shap nltk tqdm matplotlib seaborn scikit-learn
```

---

## Quick Start

### Step 1: Build XLex (One-time setup)

Generate the explainable lexicon using SHAP:

```bash
python xlex/build_xlex.py
```

This will:
* Load dataset (2264 samples)
* Run SHAP explanations on transformer
* Extract word-level importance scores
* Build and save `xlex.json`

**Note:** SHAP is computationally expensive. First run takes ~5-10 minutes. You can adjust `XLEX_BUILD_SAMPLES` in `config.py` for faster testing.

### Step 2: Train Model

```bash
python main.py
```

This will:
* Load dataset and XLex
* Create DataLoaders
* Train EnhancedFinSentiBERT model
* Evaluate on same dataset
* Save visualizations and metrics
* Save model checkpoint

**Output:**
```
results/
├── loss_curve.png          # Training loss over epochs
├── confusion_matrix.png    # Prediction vs ground truth
├── metrics.png             # Accuracy, precision, recall, F1
└── metrics.json            # Metrics in JSON format

models_checkpoint/
└── best_model.pt          # Trained model
```

### Step 3: Run Inference

```bash
python inference.py
```

For custom predictions:
```python
from inference import FinancialSentimentClassifier

classifier = FinancialSentimentClassifier(
    model_path="models_checkpoint/best_model.pt"
)

# Single prediction
result = classifier.predict(
    "The company reported strong earnings growth.",
    return_probabilities=True
)
print(result)
# Output: {
#   'text': '...',
#   'label': 'positive',
#   'confidence': 0.95,
#   'probabilities': {'positive': 0.95, 'negative': 0.03, 'neutral': 0.02}
# }

# Batch predictions
results = classifier.predict_batch([
    "Earnings increased 20% YoY.",
    "Stock price fell sharply.",
    "Results were as expected."
])
```

---

## Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 3

# Training
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5

# XLex
XLEX_PATH = "xlex.json"
XLEX_DIM = 6
MIN_WORD_FREQ = 5
XLEX_BUILD_SAMPLES = 3000  # Reduce for faster testing
```

---

## Output & Results

### Plots (saved to `results/`)

1. **loss_curve.png** - Training loss progression
2. **confusion_matrix.png** - Classification performance breakdown
3. **metrics.png** - Bar chart of all metrics

### Metrics (saved to `results/metrics.json`)

```json
{
  "accuracy": 0.8734,
  "precision": 0.8621,
  "recall": 0.8512,
  "f1": 0.8566
}
```

---

## Architecture

### EnhancedFinSentiBERT

```
Input Text
    ↓
BERT Encoder (768 dims)
    ↓
    ├─→ BERT Branch  ────→ [CLS] Token
    ├─→ XLex Branch  ────→ Attention + [CLS]
    └─→ Neutral Branch ──→ Attention + [CLS]
         (FNN + Attention)
    ↓
Concatenate (3 x 768 = 2304 dims)
    ↓
Dense Layers (2304 → 512 → 3 classes)
    ↓
Classification Output
```

### XLex

XLex (Explainable Lexicon) is a word-level importance score dictionary built using SHAP:

- **avg_importance**: Mean absolute SHAP value
- **total_importance**: Sum of absolute SHAP values
- **max_score**: Maximum SHAP value
- **min_score**: Minimum SHAP value
- **frequency**: Number of occurrences
- **polarity**: Direction (1 for positive, -1 for negative)

---

## Training Notes

- **First XLex build** (~5-10 min): SHAP explanations are expensive
- **Model training** (~2-5 min): Depends on dataset size and hardware
- **GPU recommended**: Training on CPU is significantly slower
- **Resume from checkpoint**: Load previous `models_checkpoint/best_model.pt`

### Optimization Tips

1. **Reduce samples** for faster iteration:
   ```python
   XLEX_BUILD_SAMPLES = 500  # Default: 3000
   BATCH_SIZE = 32           # Default: 16
   EPOCHS = 3                # Default: 5
   ```

2. **Use full dataset** for best accuracy:
   ```python
   XLEX_BUILD_SAMPLES = 2264  # All samples
   EPOCHS = 10+
   ```

---

## Troubleshooting

### "xlex.json not found"
```bash
python xlex/build_xlex.py
```

### CUDA out of memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `XLEX_BUILD_SAMPLES`

### Slow training on CPU
- Ensure GPU is available: `DEVICE = "cuda"` in `config.py`
- Check: `torch.cuda.is_available()`

### Model not improving
- Increase `EPOCHS`
- Adjust `LR` (learning rate)
- Use full dataset

---

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Global settings and hyperparameters |
| `main.py` | Main training entry point |
| `inference.py` | Production inference wrapper |
| `models/model.py` | EnhancedFinSentiBERT architecture |
| `xlex/build_xlex.py` | XLex generation using SHAP |
| `training/train.py` | Training loop with checkpointing |
| `training/evaluate.py` | Evaluation and metrics calculation |
| `utils/plots.py` | Visualization functions |
| `utils/metrics.py` | Metrics utilities |

---

## Pipeline Summary

```
Raw Data
    ↓
[1] Build XLex (xlex/build_xlex.py)
    ↓ Creates: xlex.json
[2] Load & Tokenize (data/dataset.py)
    ↓ Input: Sentences_AllAgree.txt
[3] Train Model (main.py → training/train.py)
    ↓ Output: models_checkpoint/best_model.pt
[4] Evaluate (training/evaluate.py)
    ↓ Metrics: accuracy, precision, recall, F1
[5] Visualize (utils/plots.py)
    ↓ Output: results/ (png + json)
[6] Inference (inference.py)
    ↓ Input: Any text
    ↓ Output: Sentiment + Confidence
```

---

## Results & Performance

Expected performance on Financial PhraseBank (after training):
- **Accuracy**: ~85%+
- **Precision**: ~84%+
- **Recall**: ~84%+
- **F1-Score**: ~84%+

*Actual results depend on hyperparameters, dataset splits, and initialization.*

---

## Dependencies

- **PyTorch** ≥2.11.0 - Deep learning framework
- **Transformers** ≥5.5.0 - BERT/DistilBERT models
- **SHAP** ≥0.51.0 - Model explainability
- **Scikit-learn** ≥1.8.0 - Metrics calculation
- **NLTK** ≥3.9.4 - Text preprocessing
- **Matplotlib** ≥3.10.8 - Visualization
- **Seaborn** ≥0.13.2 - Statistical plots
- **TQDM** ≥4.67.3 - Progress bars

---

## License

See `FinancialPhraseBank-v1.0/License.txt` for dataset license.

---

## References

- BERT: https://arxiv.org/abs/1810.04805
- SHAP: https://arxiv.org/abs/1705.07874
- Financial PhraseBank: https://www.researchgate.net/publication/251231107_FinancialPhraseBank-v10
