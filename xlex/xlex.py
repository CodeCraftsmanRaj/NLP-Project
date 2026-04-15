import json
import torch
from config import XLEX_DIM, MAX_LEN

class XLex:
    def __init__(self, path):
        with open(path) as f:
            self.lex = json.load(f)

    def encode(self, tokens):
        mat = torch.zeros(MAX_LEN, XLEX_DIM)

        for i, t in enumerate(tokens[:MAX_LEN]):
            if t in self.lex:
                mat[i] = torch.tensor(self.lex[t])

        return mat