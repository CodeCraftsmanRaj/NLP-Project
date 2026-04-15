import os

def load_phrasebank():
    texts = []
    labels = []

    label_map = {
        "positive": 0,
        "negative": 1,
        "neutral": 2
    }

    path = "FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()

            if not line or "@" not in line:
                continue

            text, label = line.rsplit("@", 1)

            label = label.strip().lower()

            if label not in label_map:
                continue

            texts.append(text.strip())
            labels.append(label_map[label])

    print(f"Loaded {len(texts)} samples")

    return texts, labels