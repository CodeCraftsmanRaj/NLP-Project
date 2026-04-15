import torch
import torch.nn as nn
from transformers import BertModel
from config import MODEL_NAME, NUM_CLASSES

class EnhancedFinSentiBERT(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.lex_linear = nn.Linear(6, 768)
        self.lex_attn = nn.MultiheadAttention(768, 4, batch_first=True)

        self.neutral_fc = nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,768),
            nn.ReLU()
        )
        self.neutral_attn = nn.MultiheadAttention(768,4,batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(768*3,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,NUM_CLASSES)
        )

    def forward(self, input_ids, mask, lex):
        bert_out = self.bert(input_ids, mask).last_hidden_state

        lex = self.lex_linear(lex)
        lex_out,_ = self.lex_attn(lex,lex,lex)

        neutral = self.neutral_fc(bert_out)
        neutral_out,_ = self.neutral_attn(neutral,neutral,neutral)

        fused = torch.cat([
            bert_out[:,0],
            lex_out[:,0],
            neutral_out[:,0]
        ], dim=1)

        return self.fc(fused)