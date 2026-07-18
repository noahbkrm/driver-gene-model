import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM, BATCH

class SNVMask(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.snv_mask_embedding = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, snv_tokens: torch.Tensor):
        mask_snv_tokens = self.snv_mask_embedding.expand_as(snv_tokens) # Broadcast to (batch, n_genes, hidden_dim)
        return mask_snv_tokens