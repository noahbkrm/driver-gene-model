import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH
from dataclasses import dataclass
from fusion import TokenEmbedding

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, batch_size: int = BATCH):
        super().__init__()
        self.query = nn.Parameter(torch.randn(batch_size, 1, hidden_dim)) # q, "What information is useful for summarizing this patient?"
        self.Wq = nn.Parameter(torch.randn(768, 768))
        self.Wk = nn.Parameter(torch.randn(768, 768))
        self.Wv = nn.Parameter(torch.randn(768, 768))
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_tokens): # input token embedding dims: (batch, n_tokens, hidden_dim)
        Q = torch.matmul(input_tokens, self.Wq) # Q: q*Wq    dims: (batch, 1, hidden_dim)
        K = torch.matmul(input_tokens, self.Wk) # K: x*Wk    dims: (batch, n_tokens, hidden_dim)
        V = torch.matmul(input_tokens, self.Wv) # V: x*Wv    dims: (batch, n_tokens, hidden_dim)
        sim_matrix = torch.matmul(Q, torch.transpose(K, -2, -1))
        # sim_matrix = Q*K.T  dims: (batch, 1, n_tokens) (K.T: batch, hidden_dim, n_tokens)

        alpha = torch.softmax(sim_matrix, -1) # dims: (batch, 1, n_tokens)
        pool_emb = torch.bmm(alpha, V) # output: a*V   dims: (batch, 1, hidden_dim)
        return self.layernorm(pool_emb.squeeze())
        