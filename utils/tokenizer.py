from transformers import BertTokenizer
from config import MODEL_NAME, MAX_LEN

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    tokens = tokenizer.tokenize(text)

    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), tokens