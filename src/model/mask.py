import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM, BATCH
from snv_encoder import SNVEmbedding

def replaceSNV()
    return True

class MaskEmbedding(nn.Module):
    def __init__(self, snv_emb: torch.Tensor):
        super().__init__()
        self.snv_mask_embedding = nn.Embedding(snv_emb.shape[0], snv_emb.shape[1], snv_emb.shape[2])