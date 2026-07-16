"""
CNV encoder - arm A (discrete embedding) of the two-arm study

Contract
--------
Turns per-patient gene-level copy-number variation calls into a set of
per-gene token vectors that live in the same 768-dim space as the clinical /
RNA / SNV tokens, allowing the shared attention-pooling module to treat all
modalities uniformly.

Input
-----
Per-patient gene-level CNV calls from `cohort.cnv`.

Each value represents the copy-number state of a gene:

    -2 : deep deletion
    -1 : deletion
     0 : neutral copy number
    +1 : gain
    +2 : amplification

The synthetic dataset represents CNVs as a patient-by-gene matrix. Although
real CNV data is naturally segment-based, this encoder uses a simplified
gene-level representation where each gene receives one token.

Output shape
------------
(batch, n_genes, HIDDEN_DIM). One token per gene.

CNV state embedding
-------------------
Each copy-number state is treated as a categorical variable.

The raw CNV values are mapped to embedding indices:

    -2 -> 0
    -1 -> 1
     0 -> 2
    +1 -> 3
    +2 -> 4
    NA -> 5

A learned `nn.Embedding(6, HIDDEN_DIM)` table converts each CNV state into
a dense representation, including a dedicated missing-value state.

Gene identity
-------------
A learned `nn.Embedding(n_genes, HIDDEN_DIM)` table provides a
patient-independent representation of each gene, analogous to positional
embeddings in a transformer.

The final token for gene `i` in patient `p` is:

    cnv_state_emb[p, i] + gene_emb[i]

This allows the model to distinguish between:

    EGFR amplification
    TP53 deletion

even if both have the same copy-number state.

Missing values
--------------
Individual CNV calls may be missing. Missing values are represented as a
sixth categorical state in the CNV embedding table.

This allows the model to learn a distinct representation for missing CNV
information rather than incorrectly treating missing values as neutral copy
number.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM
from dataclasses import dataclass

class CNVEmbedding(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.cnv_state_embedding = nn.Embedding(6, hidden_dim) # -2, -1, 0, 1, 2, NA
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.register_buffer(  # Static so it won't be treated as a parameter to be optimized
            "gene_ids",
            torch.arange(n_genes)
        )
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> torch.Tensor:

        df = df.copy()

        df = df + 2 # Shift all values to be >= 0
        df = df.fillna(5).astype("int64") # fill missing values as 6th item (index 5)
        cnv_state_tensor = torch.from_numpy(df.to_numpy(dtype=np.int64))
        return cnv_state_tensor
    
    def forward(self, state_tensor):

        gene_emb = self.gene_embedding(self.gene_ids)
        state_emb = self.cnv_state_embedding(state_tensor)

        combined_emb = gene_emb + state_emb
        return self.layernorm(combined_emb)

if __name__ == "__main__":

    df = load_cohort().cnv

    # Prepare CNV states
    states = CNVEmbedding.prepare(df)

    print("Full tensor:")
    print(states.shape)
    print(states.dtype)
    print(states.min())
    print(states.max())

    # Create DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(states)

    batch_size = 32
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    model = CNVEmbedding(
        n_genes=df.shape[1]
    )

    # Test first batch
    for batch_idx, (cnv_batch,) in enumerate(loader):

        print("\nBatch:")
        print("Input:", cnv_batch.shape)

        output = model(cnv_batch)

        print("Output:", output.shape)

        # Only test one batch
        break