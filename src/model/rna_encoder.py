"""RNA encoder - arm A (discrete embedding) of the two-arm study

Contract
--------
Turns per-patient bulk RNA counts into a per-gene set of token vectors that
sit in the same 768-dim space as the clinical / SNV / CNV tokens, so the
shard attention-pooling module can treat them uniformly. 

Gene filter
-----------
Keep a gene only if it has `count >= 1` in at least 5% of TRAINING patients. 
The surviving gene list is fit once o nthe trainig split then reused for val/test. 

Output shape
----------
(batch, n_kept_genes, HIDDEN_DIM). One token per surviving gene. 

Value embedding
---------------
For each (patient, gene) count `x`:
    x = x / total_counts_for_patients * 1e6      # library-size normalize -> CPM
    x = log1p(x)                                 # log transform
    x = (x - train_gene_mean) / train_gene_std   # per-gene z-scroe (train-fit)
Then project each scalar through a shared `nn.Linear(1, HIDDEN_DIM)`. 

Gene identity
------------
A learned `nn.Embedding(n_kept_genes, HIDDEN_DIM)` table. Row `i` is the 
patient-independent representation of gene `i`, analogous to positional
embeddings in a transformer. The final token for gene `i` in patient `p`
is `value_emb[p, i] + gene_emb[i]`. 

Missing values
--------------
Individual count NAs are imputed with the TRAINING gene mean (in the same
normalized space) and flagged in a value mask. Same pattern as clinical:
a learned "was-missing" bias vector added to the projected value when the
mask says missing. Missingness is per-cell, not per-gene, so the bias is
shared across genes. 
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM
from dataclasses import dataclass

@dataclass
class RnaStats:
    kept_gene_names: list[str]
    gene_mean: np.ndarray
    gene_std: np.ndarray

class RnaEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, gene_count: int = 2000):
        super().__init__()
        self.expression = nn.Embedding(gene_count, hidden_dim)
        self.missing_bias = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal(self.missing_bias, std = 0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    @staticmethod
    def prepare(
        df: pd.DataFrame,
        stats: RnaStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        df = df.copy()

        observed_mask = 1 - df.isna() # True (1) when false, so 1- means 0 when false

        # Drop gene if fewer than 5% of columns are populated
        mask = df.isna() | (df == 0) | (df == "0")
        valid_fraction = 1 - mask.mean(axis=0)
        df2 = df2.loc[:, valid_fraction >= 0.05]



        observed_mask = torch.from_numpy(observed_mask.to_numpy(dtype = np.float32))

        return expression_tensor, observed_mask
    
    def forward(self, expression_tensor, mask):
        return True

if __name__ == "__main__":
    df = load_cohort().rna
    df = df.copy()
    df2 = df.copy()
    print(df.shape)

    # Drop gene if fewer than 5% of columns are populated
    mask = df.isna() | (df == 0) | (df == "0")
    valid_fraction = 1 - mask.mean(axis=0)
    df2 = df2.loc[:, valid_fraction >= 0.05]

    print(df2.shape)