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
    train_gene_mean: pd.Series
    train_gene_std: pd.Series

    @property
    def n_genes(self):
        return len(self.kept_gene_names)

class RnaEmbedding(nn.Module):
    def __init__(self,  stats: RnaStats, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.expression = nn.Linear(1, hidden_dim)
        self.missing_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.missing_bias, std = 0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gene_embedding = nn.Embedding(stats.n_genes, hidden_dim)
        self.register_buffer(  # Static so it won't be treated as a parameter to be optimized
        "gene_ids",
        torch.arange(stats.n_genes)
        )
    
    @staticmethod
    def fit(train_df: pd.DataFrame) -> RnaStats: # Fitting RnaStats on training data

        df = train_df.copy()
        # Drop gene if fewer than 5% of columns are populated
        mask = df.isna() | (df == 0)
        valid_fraction = 1 - mask.mean(axis=0)
        df = df.loc[:, valid_fraction >= 0.05]
        col_names = df.columns.tolist()
        
        # Normalize for library size using CPM (Counts Per Million)
        library_size = df.sum(axis = 1)
        df = df.div(library_size, axis = 0) * 1e6

        # Perform log1p normalization
        df = np.log1p(df)

        # Compute mean and standard deviation as np.ndarray
        train_gene_std = df.std(axis = 0, ddof = 0)
        train_gene_mean = df.mean(axis = 0)

        rna_stats_obj = RnaStats(kept_gene_names=col_names, train_gene_mean= train_gene_mean, train_gene_std=train_gene_std)
        return rna_stats_obj

    @staticmethod
    def prepare(
        df: pd.DataFrame,
        stats: RnaStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        df = df.copy()

        df = df[stats.kept_gene_names] # Filter to training retained genes
        observed_mask = 1 - df.isna() # True (1) when false, so 1- means 0 when false

        # Normalize for library size using CPM (Counts Per Million)
        library_size = df.sum(axis = 1)
        df = df.div(library_size, axis = 0) * 1e6

        # Perform log1p normalization
        df = np.log1p(df)

         # Fill NAs
        df = df.fillna(stats.train_gene_mean)

        # Calculate z-score using training means in standard deviation
        df = (df - stats.train_gene_mean) / stats.train_gene_std

        expression_tensor = torch.from_numpy(df.to_numpy(dtype = np.float32))
        observed_mask = torch.from_numpy(observed_mask.to_numpy(dtype = np.float32))

        return expression_tensor, observed_mask
    
    def forward(self, expression_tensor, mask):

        gene_emb = self.gene_embedding(self.gene_ids)

        expression_emb = self.expression(expression_tensor.unsqueeze(-1))
        was_missing_exp = (1 - mask).unsqueeze(-1)
        expression_emb = expression_emb + was_missing_exp * self.missing_bias

        combined_emb = gene_emb + expression_emb # Tokens
        return self.layernorm(combined_emb)

if __name__ == "__main__":
    # Load synthetic RNA counts
    df = load_cohort().rna

    # Fit preprocessing statistics (normally train split only)
    stats = RnaEmbedding.fit(df)

    print(f"Retained genes: {stats.n_genes}")

    # Convert dataframe -> tensors
    expression_tensor, mask = RnaEmbedding.prepare(df, stats)

    print(expression_tensor.shape, expression_tensor.dtype)
    print(mask.shape, mask.dtype)

    # Build encoder
    model = RnaEmbedding(stats)

    # Forward pass
    batch_size = 32

    for i in range(0, len(df), batch_size):
        batch = expression_tensor[i:i+batch_size]
        mask_batch = mask[i:i+batch_size]

        out = model(batch, mask_batch)

        print(out.shape)