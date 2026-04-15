# 🚀 How to Run the Project & Get Results

## Complete Step-by-Step Guide

---

## ⚙️ Prerequisites

Make sure your virtual environment is activated:

```bash
cd /home/raj_99/Projects/NLP_project/nlp-project
source .venv/bin/activate
```

Verify activation:
```bash
which python
# Should show: /home/raj_99/Projects/NLP_project/nlp-project/.venv/bin/python
```

---

## 📋 3-Step Pipeline

### **STEP 1️⃣: Build XLex (One-time)**

This generates the explainable lexicon using SHAP. Takes ~5-10 minutes.

```bash
python xlex/build_xlex.py
```

**What it does:**
- Loads Financial PhraseBank dataset (2264 samples)
- Runs SHAP explanations on BERT model
- Extracts word-level importance scores
- Builds lexicon with sentiment values

**Expected output:**
```
Loaded 2264 samples
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 372.82it/s]
[...BERT load report...]
XLex built: 1234 -> xlex.json
```

**Output file:**
- ✓ `xlex.json` (created in project root)

---

### **STEP 2️⃣: Train Model**

This trains the EnhancedFinSentiBERT model. Takes ~3-5 minutes.

```bash
python main.py
```

**What it does:**
1. Loads dataset & XLex
2. Creates DataLoaders
3. Trains model for 5 epochs (configurable in `config.py`)
4. Evaluates on same dataset
5. Generates visualizations
6. Saves model checkpoint
7. Exports metrics to JSON

**Expected output:**
```
============================================================
Financial Sentiment Analysis Pipeline
============================================================

[1/5] Loading dataset...
  ✓ Loaded 2264 samples

[2/5] Loading XLex...
  ✓ XLex loaded (xlex.json)

[3/5] Preparing data loaders...
  ✓ DataLoader created (batch_size=16)

[4/5] Training model...
  ✓ Model initialized (device=cpu/cuda)
Epoch 1/5: 100%|██████████| 142/142 [01:23<00:00, 0.59it/s]
Epoch 1/5 - Avg Loss: 0.8234
Epoch 2/5: 100%|██████████| 142/142 [01:22<00:00, 0.60it/s]
Epoch 2/5 - Avg Loss: 0.6891
...
[5/5] Evaluating model...
Evaluating: 100%|██████████| 142/142 [00:45<00:00, 3.14it/s]

Evaluation Results:
  Accuracy: 0.8734
  Precision: 0.8621
  Recall: 0.8512
  F1: 0.8566

==================================================
Model Performance Summary
==================================================
Accuracy       : 0.8734
Precision      : 0.8621
Recall         : 0.8512
F1             : 0.8566
==================================================

✓ Loss plot saved: results/loss_curve.png
✓ Confusion matrix saved: results/confusion_matrix.png
✓ Metrics plot saved: results/metrics.png
✓ Metrics saved: results/metrics.json
✓ Checkpoint saved: models_checkpoint/best_model.pt

============================================================
Training pipeline completed successfully!
Results saved to: results/
============================================================
```

**Output files created:**
```
results/
├── loss_curve.png              # Training loss over epochs
├── confusion_matrix.png        # Classification matrix heatmap
├── metrics.png                 # Metrics bar chart
└── metrics.json               # Metrics in JSON format

models_checkpoint/
└── best_model.pt              # Trained model
```

---

### **STEP 3️⃣: Run Inference**

Use the trained model for predictions.

**Option A: Run demo predictions**

```bash
python inference.py
```

**Expected output:**
```
============================================================
Financial Sentiment Classification - Inference
============================================================

Testing predictions:

Text: The company reported strong earnings growth this quarter.
Prediction: POSITIVE (confidence: 92.45%)
Probabilities: {'positive': 0.9245, 'negative': 0.0312, 'neutral': 0.0443}
------------------------------------------------------------
Text: Stock prices fell significantly due to market concerns.
Prediction: NEGATIVE (confidence: 87.62%)
Probabilities: {'positive': 0.0234, 'negative': 0.8762, 'neutral': 0.1004}
------------------------------------------------------------
Text: The results were within expected ranges.
Prediction: NEUTRAL (confidence: 78.91%)
Probabilities: {'positive': 0.1856, 'negative': 0.0253, 'neutral': 0.7891}
------------------------------------------------------------
```

**Option B: Use in Python code**

```python
from inference import FinancialSentimentClassifier

# Initialize classifier
classifier = FinancialSentimentClassifier()

# Single prediction
text = "Earnings increased significantly this quarter"
result = classifier.predict(text, return_probabilities=True)
print(result)
# Output:
# {
#   'text': 'Earnings increased significantly this quarter',
#   'label': 'positive',
#   'confidence': 0.9234,
#   'label_id': 0,
#   'probabilities': {
#     'positive': 0.9234,
#     'negative': 0.0412,
#     'neutral': 0.0354
#   }
# }

# Batch predictions
texts = [
    "Strong earnings growth",
    "Market downturn concerns",
    "Revenue steady"
]
results = classifier.predict_batch(texts, return_probabilities=True)
for r in results:
    print(f"{r['text']} → {r['label']} ({r['confidence']:.1%})")
```

---

## 📊 View Final Results

### **1. View Plots**

```bash
# Loss curve
open results/loss_curve.png        # Mac
xdg-open results/loss_curve.png    # Linux
start results/loss_curve.png       # Windows

# Confusion matrix
open results/confusion_matrix.png

# Metrics chart
open results/metrics.png
```

### **2. View Metrics JSON**

```bash
cat results/metrics.json
```

**Output:**
```json
{
  "accuracy": 0.8734,
  "precision": 0.8621,
  "recall": 0.8512,
  "f1": 0.8566
}
```

### **3. View Model Checkpoint**

```bash
ls -lh models_checkpoint/best_model.pt
# Shows model file size
```

---

## 🔧 Customization

### Adjust Training Parameters

Edit `config.py`:

```python
# For faster testing (reduce time)
XLEX_BUILD_SAMPLES = 500   # Default: 3000
BATCH_SIZE = 32            # Default: 16
EPOCHS = 3                 # Default: 5

# For better accuracy (takes longer)
XLEX_BUILD_SAMPLES = 2264  # All samples
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5                  # Lower learning rate
```

Then run:
```bash
python main.py
```

---

## 📁 Complete Output Structure

After running all steps:

```
nlp-project/
├── xlex.json                    # Generated lexicon
│
├── results/                     # 📊 MAIN OUTPUTS
│   ├── loss_curve.png          # Training progress chart
│   ├── confusion_matrix.png    # Classification breakdown
│   ├── metrics.png             # Performance metrics chart
│   └── metrics.json            # Metrics data (machine-readable)
│
├── models_checkpoint/          # 🤖 MODEL
│   └── best_model.pt           # Trained model
│
├── logs/                       # 📝 Logs (future)
│
├── config.py
├── main.py
├── inference.py
├── README.md
└── [other project files]
```

---

## ✅ Verification Checklist

After running `python main.py`, verify:

- [ ] `xlex.json` exists
- [ ] `results/loss_curve.png` exists
- [ ] `results/confusion_matrix.png` exists
- [ ] `results/metrics.png` exists
- [ ] `results/metrics.json` exists
- [ ] `models_checkpoint/best_model.pt` exists
- [ ] Console shows metrics (accuracy, precision, recall, F1)

---

## 🐛 Troubleshooting

### Error: "xlex.json not found"
```bash
python xlex/build_xlex.py
```

### Error: "FileNotFoundError: Sentences_AllAgree.txt"
Ensure dataset exists:
```bash
ls FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
```

### GPU Out of Memory
```python
# In config.py, reduce:
BATCH_SIZE = 8  # From 16
XLEX_BUILD_SAMPLES = 1000  # From 3000
```

### CPU Too Slow
```bash
# Check GPU availability:
python -c "import torch; print(torch.cuda.is_available())"
# If True, it's using GPU. If False, increase BATCH_SIZE for CPU efficiency
```

### Model Predictions All Same Label
This usually means:
1. Model needs more epochs (`EPOCHS = 10`)
2. Dataset might be imbalanced
3. Try different `LR` value

---

## 📈 Expected Performance

| Metric | Expected |
|--------|----------|
| Accuracy | 82-88% |
| Precision | 81-87% |
| Recall | 80-86% |
| F1-Score | 80-86% |

*Actual results vary based on:*
- Hardware (GPU vs CPU)
- Random initialization
- Dataset order
- Hyperparameters

---

## 🎯 Quick Command Reference

```bash
# Activate environment
source .venv/bin/activate

# Step 1: Build XLex (5-10 min)
python xlex/build_xlex.py

# Step 2: Train Model (3-5 min)
python main.py

# Step 3: Run Inference
python inference.py

# View results
cat results/metrics.json
open results/loss_curve.png
open results/confusion_matrix.png
open results/metrics.png
```

---

## 💡 Tips

1. **First run is slow** - SHAP takes time, subsequent runs are faster
2. **GPU recommended** - Training is 5-10x faster on GPU
3. **Monitor progress** - Use TQDM progress bars to see real-time updates
4. **Save checkpoints** - Model automatically saved for resumable training
5. **Export metrics** - JSON format lets you track experiments

---

## 📞 Need Help?

Refer to:
- [README.md](README.md) - Full documentation
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Architecture details
- Docstrings in Python files - Detailed API docs

---

**You're all set! 🎉 Happy training!**
