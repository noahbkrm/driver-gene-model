import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM, BATCH

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from clinical_encoder import ClinicalEmbedding
from rna_encoder import RnaEmbedding, RnaStats
from fusion import TokenEmbedding
from attention_pooling import AttentionPooling
from mask import SNVMask

class PatientModel(nn.Module):
    def __init__(self, rna_stats: RnaStats, n_genes: int, n_variant_genes: int, hidden_dim: int = HIDDEN_DIM, batch_size: int = BATCH):
        super().__init__()
        self.cnv_encoder =  CNVEmbedding(n_variant_genes, hidden_dim)
        self.snv_encoder = SNVEmbedding(n_variant_genes, hidden_dim)
        self.snv_masker = SNVMask(hidden_dim)
        self.clinical_encoder = ClinicalEmbedding(hidden_dim)
        self.rna_encoder = RnaEmbedding(rna_stats, hidden_dim)
        self.combine_tokens = TokenEmbedding(hidden_dim)
        self.attention_pooling = AttentionPooling(hidden_dim)

    def forward(self, batch, mask_snv = False):
        clinical_tokens = self.clinical_encoder(
            batch["clinical_cat"],
            batch["clinical_cont"],
            batch["clinical_mask"],
        )

        rna_tokens = self.rna_encoder(
            batch["rna_expression"],
            batch["rna_mask"],
        )

        cnv_tokens = self.cnv_encoder(
            batch["cnv_states"],
        )

        snv_tokens = self.snv_encoder(
            batch["snv_states"],
        )

        if mask_snv:
            snv_tokens = self.snv_masker(snv_tokens)

        token_emb = self.combine_tokens(clinical_tokens, rna_tokens, cnv_tokens, snv_tokens)
        patient_emb = self.attention_pooling(token_emb)

        return patient_emb