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

# Add type suggestions
def prepare_model_inputs(
        clinical_df: pd.DataFrame, 
        snv_df: pd.DataFrame, 
        cnv_df: pd.DataFrame,
        rna_df: pd.DataFrame, 
        train_means: dict[str,float], 
        rna_stats: RnaStats
    ) -> dict[str, torch.Tensor]:

    clinical_cat, clinical_cont, clinical_mask = ClinicalEmbedding.prepare(clinical_df, train_means)
    rna_expression, rna_mask = RnaEmbedding.prepare(rna_df, rna_stats)
    cnv_states = CNVEmbedding.prepare(cnv_df)
    snv_states = SNVEmbedding.prepare(snv_df)
    batch = {
        "clinical_cat": clinical_cat,
        "clinical_cont": clinical_cont,
        "clinical_mask": clinical_mask,
        "rna_expression": rna_expression,
        "rna_mask": rna_mask,
        "cnv_states": cnv_states,
        "snv_states": snv_states,
    }
    return batch

class PatientModel(nn.Module):
    def __init__(self, rna_stats: RnaStats, n_genes: int, hidden_dim: int = HIDDEN_DIM, batch_size: int = BATCH):
        super().__init__()
        self.cnv_encoder =  CNVEmbedding(n_genes, hidden_dim)
        self.snv_encoder = SNVEmbedding(n_genes, hidden_dim)
        self.clinical_encoder = ClinicalEmbedding(hidden_dim)
        self.rna_encoder = RnaEmbedding(rna_stats, hidden_dim)
        self.combine_tokens = TokenEmbedding(hidden_dim)
        self.attention_pooling = AttentionPooling(hidden_dim, batch_size)

    def forward(self, batch):
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

        token_emb = self.combine_tokens(clinical_tokens, rna_tokens, cnv_tokens, snv_tokens)
        patient_emb = self.attention_pooling(token_emb)

        return patient_emb