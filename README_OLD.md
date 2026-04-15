# Financial Sentiment Analysis with XLex + EnhancedFinSentiBERT

## Overview

This project implements a financial sentiment analysis system based on:

* **EnhancedFinSentiBERT architecture** (multi-branch transformer model)
* **XLex (Explainable Lexicon)** generated using SHAP

The model combines:

1. **BERT semantic understanding**
2. **Explainable lexicon features (XLex)**
3. **Neutral sentiment feature extraction**

The goal is to improve sentiment classification in financial text, especially for **neutral and subtle sentiment expressions**.

---

## Features

* Multi-branch neural architecture:

  * BERT branch (semantic features)
  * Lexicon branch (XLex features)
  * Neutral feature branch
* Explainable lexicon generation using SHAP
* Training and evaluation pipeline
* Visualization:

  * Training loss
  * Confusion matrix

---

## Project Structure

```
fin_senti_xlex/
│
├── config.py
├── main.py
│
├── data/
│   ├── load_data.py
│   ├── dataset.py
│
├── models/
│   ├── model.py
│
├── xlex/
│   ├── build_xlex.py
│   ├── xlex.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│
├── utils/
│   ├── tokenizer.py
│   ├── plots.py
│
└── FinancialPhraseBank-v1.0/
```

---

## Dataset

This project uses the **Financial PhraseBank** dataset.

### Required file

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

### Create environment

```
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```
pip install torch transformers shap nltk tqdm matplotlib seaborn scikit-learn
```

---

## Step 1 — Build XLex

Generate the explainable lexicon using SHAP:

```
python xlex/build_xlex.py
```

This will:

* Load dataset
* Run SHAP on transformer
* Extract word-level importance
* Build lexicon
* Save:

```
xlex.json
```

---

## Step 2 — Train Model

```
python main.py
```

This will:

* Load dataset
* Load XLex
* Train model
* Evaluate performance
* Show plots

---

## Output

### Metrics

* Accuracy
* Precision
* Recall
* F1-score

### Visualizations

* Training loss curve
* Confusion matrix

---

## Notes

* XLex generation is **slow** (SHAP is expensive)

* You can reduce samples in `config.py`:

  ```
  XLEX_BUILD_SAMPLES = 1000
  ```

* For better performance:

  * Use full dataset
  * Train more epochs

---

## Run Order

```
python xlex/build_xlex.py
python main.py
```

---

## Summary

This project implements a hybrid approach combining:

* Deep learning (transformers)
* Explainable AI (SHAP)
* Domain knowledge (financial lexicons)

Result: improved financial sentiment analysis with interpretability.
# NLP-Project
