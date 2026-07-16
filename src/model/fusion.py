# Main block with merging token, multi-head attention pooling (1-head), query pooling
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM
from dataclasses import dataclass

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from clinical_encoder import ClinicalEmbedding
from rna_encoder import RnaEmbedding
    
class TokenEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        self.register_buffer(  # Static so it won't be treated as a parameter to be optimized
            "modality_embedding",
            nn.Embedding(4, hidden_dim)
        )
    
    def forward(
            self,
            clinical_tokens: torch.Tensor,
            rna_tokens: torch.Tensor,
            cnv_tokens: torch.Tensor,
            snv_tokens: torch.Tensor,
        ):

        clinical_tokens = clinical_tokens + self.modality_embedding
        rna_tokens = rna_tokens + self.modality_embedding
        cnv_tokens = cnv_tokens + self.modality_embedding
        snv_tokens = snv_tokens + self.modality_embedding

        combined_tokens = torch.cat([clinical_tokens, rna_tokens, cnv_tokens, snv_tokens], dim = 1)
        return combined_tokens

