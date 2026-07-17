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
from rna_encoder import RnaEmbedding, RnaStats
from fusion import TokenEmbedding
from attention_pooling import AttentionPooling

# Add type suggestions
def prepare_model_inputs(clinical_df, snv_df, cnv_df, rna_df, train_means, rna_stats: RnaStats):
    clinical_cat, clinical_cont, clinical_mask = ClinicalEmbedding.prepare(clinical_df, train_means)
    rna_expression, rna_mask = RnaEmbedding.prepare(rna_df, rna_stats)
    cnv_states = CNVEmbedding.prepare(cnv_df)
    snv_states = SNVEmbedding.prepare(snv_df)
    batch_dict = {
        "clinical_cat": clinical_cat,
        "clinical_cont": clinical_cont,
        "clinical_mask": clinical_mask,
        "rna_expression": rna_expression,
        "rna_mask": rna_mask,
        "cnv_states": cnv_states,
        "snv_states": snv_states,
    }
    return batch_dict

class PatientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnv_encoder =  CNVEmbedding()
        self.snv_encoder = SNVEmbedding()
        self.clinical_encoder = ClinicalEmbedding()
        self.rna_encoder = RnaEmbedding()
        self.combine_tokens = TokenEmbedding()
        self.attention_pooling = AttentionPooling()

    def forward(self, batch_dict):
        clinical_tokens = self.clinical_encoder(
            batch_dict["clinical_cat"],
            batch_dict["clinical_cont"],
            batch_dict["clinical_mask"],
        )

        rna_tokens = self.rna_encoder(
            batch_dict["rna_expression"],
            batch_dict["rna_mask"],
        )

        cnv_tokens = self.cnv_encoder(
            batch_dict["cnv_state"],
        )

        snv_tokens = self.snv_encoder(
            batch_dict["snv_state"],
        )

        token_emb = self.combine_tokens(clinical_tokens, rna_tokens, cnv_tokens, snv_tokens)
        patient_emb = self.attention_pooling(token_emb)

        return patient_emb