import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH
from dataclasses import dataclass
from fusion import TokenEmbedding

class AttentionPooling(nn.Module):
    def __init__(self, input_tokens: TokenEmbedding = input_tokens, hidden_dim: int = HIDDEN_DIM, batch_size: int = BATCH):
        super().__init__()
        self.query = nn.Parameter(torch.randn(batch_size, 1, hidden_dim)) # q
        self.Wq = nn.Parameter(torch.randn(768, 768))
        self.Wk = nn.Parameter(torch.randn(768, 768))
        self.Wv = nn.Parameter(torch.randn(768, 768))
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_tokens):
        K = torch.matmul(input_tokens, self.Wk)
        V = torch.matmul(input_tokens, self.Wv)
        att_mult = torch.matmul(self.query, torch.transpose(K, 1, 2))

        alpha = torch.softmax(att_mult, 1)
        pool_emb = torch.bmm(alpha, V)
        return pool_emb.squeeze()
        