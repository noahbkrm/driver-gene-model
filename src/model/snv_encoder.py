"""
SNV encoder - arm A (discrete embedding) of the two-arm study

Contract
--------
Turns per-patient gene-level somatic single-nucleotide variant (SNV) calls
into a set of per-gene token vectors that live in the same 768-dim space as
the clinical / RNA / CNV tokens, allowing the shared attention-pooling
module to treat all modalities uniformly.

Input
-----
Per-patient gene-level SNV calls from `cohort.snv`.

Each value represents whether a somatic mutation is present in a gene:

     0 : wild-type (no mutation)
     1 : mutation present

The synthetic dataset represents SNVs as a patient-by-gene binary matrix.
Each gene receives one token regardless of whether it is mutated.

Output shape
------------
(batch, n_genes, HIDDEN_DIM). One token per gene.

Mutation state embedding
------------------------
Each mutation state is treated as a categorical variable.

The raw SNV values are mapped to embedding indices:

     0 -> 0    (wild-type)
     1 -> 1    (mutated)

A learned `nn.Embedding(3, HIDDEN_DIM)` table converts each mutation state
into a dense representation. The third embedding index is reserved for
missing values.

Gene identity
-------------
A learned `nn.Embedding(n_genes, HIDDEN_DIM)` table provides a
patient-independent representation of each gene, analogous to positional
embeddings in a transformer.

The final token for gene `i` in patient `p` is:

    snv_state_emb[p, i] + gene_emb[i]

This allows the model to distinguish between mutations occurring in
different genes. For example:

    TP53 mutation
    KRAS mutation

receive different token representations despite sharing the same mutation
state.

Missing values
--------------
Individual SNV calls may be missing. Missing values are represented as a
third categorical state in the mutation-state embedding table.

This allows the model to learn a distinct representation for missing
mutation information rather than incorrectly assuming that a missing value
indicates a wild-type gene.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM
from dataclasses import dataclass

class SNVEmbedding(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.snv_state_embedding = nn.Embedding(3, hidden_dim) # 0, 1, NA
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.register_buffer(  # Static so it won't be treated as a parameter to be optimized
            "gene_ids",
            torch.arange(n_genes)
        )
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> torch.Tensor:

        df = df.copy()

        df = df.fillna(2).astype("int64") # fill missing values as 3rd item (index 2)
        snv_state_tensor = torch.from_numpy(df.to_numpy(dtype=np.int64))
        return snv_state_tensor
    
    def forward(self, snv_state_tensor):

        gene_emb = self.gene_embedding(self.gene_ids)
        state_emb = self.snv_state_embedding(snv_state_tensor)

        combined_emb = gene_emb + state_emb
        return self.layernorm(combined_emb)

if __name__ == "__main__":

    df = load_cohort().snv

    # Prepare SNV states
    states = SNVEmbedding.prepare(df)

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
    model = SNVEmbedding(
        n_genes=df.shape[1]
    )

    # Test first batch
    for batch_idx, (snv_batch,) in enumerate(loader):

        print("\nBatch:")
        print("Input:", snv_batch.shape)

        output = model(snv_batch)

        print("Output:", output.shape)

        # Only test one batch
        break
    print(output.type)
    print(output.shape[1])