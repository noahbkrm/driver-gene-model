import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH
from dataclasses import dataclass
from fusion import TokenEmbedding

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim)) # q, "What information is useful for summarizing this patient?"
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_tokens, hidden_dim: int = HIDDEN_DIM): # input token embedding dims: (batch, n_tokens, hidden_dim)
        B = input_tokens.size(0)
        query = self.query.expand(B,-1,-1)

        Q = self.Wq(query) # Q: q*Wq    dims: (batch, 1, hidden_dim)
        K = self.Wk(input_tokens) # K: x*Wk    dims: (batch, n_tokens, hidden_dim)
        V = self.Wv(input_tokens) # V: x*Wv    dims: (batch, n_tokens, hidden_dim)
        sim_matrix = torch.matmul(Q,torch.transpose(K,-2,-1)) / np.sqrt(hidden_dim)
        # sim_matrix = Q*K.T  dims: (batch, 1, n_tokens) (K.T: batch, hidden_dim, n_tokens)

        alpha = torch.softmax(sim_matrix, -1) # dims: (batch, 1, n_tokens)
        pool_emb = torch.bmm(alpha, V) # output: a*V   dims: (batch, 1, hidden_dim)
        return self.layernorm(pool_emb.squeeze(1))
        