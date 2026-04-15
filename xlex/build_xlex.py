import warnings

# Suppress transformers library deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import shap
import torch
import numpy as np
import json
import re
from collections import defaultdict
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import XLEX_BUILD_SAMPLES, MIN_WORD_FREQ, MODEL_NAME, NUM_CLASSES, MAX_LEN

from data.load_data import load_phrasebank
texts, _ = load_phrasebank()

nltk.download("stopwords")
nltk.download("wordnet")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_token(t):
    t = t.lower()
    t = re.sub(r"[^a-z]", "", t)
    if len(t) <= 2 or t in stop_words:
        return None
    return lemmatizer.lemmatize(t)

# dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
# texts = dataset["train"]["sentence"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
).to(DEVICE)

model.eval()

def predict_proba(batch_texts):
    encoded = tokenizer(
        list(batch_texts),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)

    return probs.detach().cpu().numpy()

explainer = shap.Explainer(predict_proba, tokenizer)

stats = defaultdict(lambda: {"sum":0,"count":0,"max":-1e9,"min":1e9})

for text in tqdm(texts[:XLEX_BUILD_SAMPLES]):
    sv = explainer([text])
    tokens = sv.data[0]
    values = sv.values[0]

    scores = np.max(values, axis=1)

    for tok, sc in zip(tokens, scores):
        tok = clean_token(tok)
        if tok is None:
            continue

        s = stats[tok]
        s["sum"] += abs(sc)
        s["count"] += 1
        s["max"] = max(s["max"], sc)
        s["min"] = min(s["min"], sc)

xlex = {}

for w, s in stats.items():
    if s["count"] < MIN_WORD_FREQ:
        continue

    avg = s["sum"]/s["count"]
    polarity = 1 if s["max"] > abs(s["min"]) else -1

    xlex[w] = [avg, s["sum"], s["max"], s["min"], s["count"], polarity]

xlex_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xlex.json")

with open(xlex_path, "w") as f:
    json.dump(xlex,f)

print("XLex built:", len(xlex), "->", xlex_path)