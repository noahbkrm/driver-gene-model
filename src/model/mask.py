import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM, BATCH

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding

def replaceSNV()
    return True

class MaskEmbedding(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        