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
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)

    def forward(
        self,
        clinical_tokens,
        rna_tokens,
        cnv_tokens,
        snv_tokens,
    ):

        clinical_ids = torch.zeros(
            clinical_tokens.shape[1],
            dtype=torch.long,
            device=clinical_tokens.device
        )

        rna_ids = torch.ones(
            rna_tokens.shape[1],
            dtype=torch.long,
            device=rna_tokens.device
        )

        cnv_ids = torch.full(
            (cnv_tokens.shape[1],),
            2,
            dtype=torch.long,
            device=cnv_tokens.device
        )

        snv_ids = torch.full(
            (snv_tokens.shape[1],),
            3,
            dtype=torch.long,
            device=snv_tokens.device
        )

        clinical_tokens = clinical_tokens + self.embedding(clinical_ids)
        rna_tokens = rna_tokens + self.embedding(rna_ids)
        cnv_tokens = cnv_tokens + self.embedding(cnv_ids)
        snv_tokens = snv_tokens + self.embedding(snv_ids)

        tokens = torch.cat(
            [
                clinical_tokens,
                rna_tokens,
                cnv_tokens,
                snv_tokens,
            ],
            dim=1
        )

        return tokens

