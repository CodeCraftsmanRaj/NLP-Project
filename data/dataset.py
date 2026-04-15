import torch
from torch.utils.data import Dataset
from utils.tokenizer import tokenize

class FinancialDataset(Dataset):
    def __init__(self, texts, labels, xlex):
        self.texts = texts
        self.labels = labels
        self.xlex = xlex

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        input_ids, attention_mask, tokens = tokenize(text)
        lex = self.xlex.encode(tokens)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lex": lex,
            "label": torch.tensor(self.labels[idx])
        }