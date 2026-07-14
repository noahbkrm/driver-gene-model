"""Clinical encoder - arm A (discrete embedding) of the two-arm study. 

Contract
--------
Turns per-patient clinical covariates into a small set of learned token
vectors that live in the same space as the RNA/SNV/CNV tokens, so the
shared attention-pooling module can treat them uniformly. 

Input columns (5, from `cohort.clinical`)
    Categorical (lookup-table embedded):
    sex     : {0, 1}
    stage   : {1, 2, 3, 4}
    site    : {0, 1, 2, 3}
Continuous (projected + missingness-aware):
    age
    tumor_purity_pct

Explicity EXCLUDED (leakage): `os_months`, `os_event`. These are outcomes
and must never enter the encoder. They stay in `cohort.clinical` for evaluation only. 

Input -> tensor conversion
--------------------------
`forward()` signature
    categorical : LongTensor (batch, 3)   # columns ordered [sex, stage, site]
    continuous  : FLoatTensor (batch, 2)  # colmns ordered [age, tumor_purity_pct]
    cont_mask   : FloatTensor (batch, 2)  # 1.0 if observed, 0 if missing

Output
    Tensor of shape (batch, 5, HIDDEN_DIM)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM

class ClinicalEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.sex = nn.Embedding(3, hidden_dim)
        self.stage = nn.Embedding(5, hidden_dim)
        self.site = nn.Embedding(5, hidden_dim)
        self.age = nn.Linear(1, hidden_dim)
        self.tumor_purity = nn.Linear(1, hidden_dim)
        self.missing_bias = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.missing_bias, std=0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    @staticmethod
    def prepare(
        df: pd.DataFrame,
        train_means: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        df = df.copy()
        observed_mask = 1-df[["age", "tumor_purity_pct"]].isna()

        df["sex"] = df["sex"].fillna(2).astype("int64")
        df["site"] = df["site"].fillna(4).astype("int64")
        df["stage"] = (df["stage"]-1).fillna(4).astype("int64")

        df["age"] = df["age"].astype("Float32").fillna(train_means["age"])
        df["tumor_purity_pct"] = df["tumor_purity_pct"].astype("Float32").fillna(train_means["tumor_purity_pct"])

        cat_cols = torch.from_numpy(df[["sex", "stage", "site"]].to_numpy(dtype=np.int64))
        cont_cols = torch.from_numpy(df[["age", "tumor_purity_pct"]].to_numpy(dtype=np.float32))
        cont_mask = torch.from_numpy(observed_mask.to_numpy(dtype=np.float32))

        return cat_cols, cont_cols, cont_mask
    
    def forward(self, categorical, continuous, cont_mask):

        sex_ids = categorical[:, 0]
        stage_ids = categorical[:, 1]
        site_ids = categorical[:, 2]
        age_vals = continuous[:, 0:1]
        tp_vals = continuous[:, 1:2]

        age_emb = self.age(age_vals)
        tumor_purity_emb = self.tumor_purity(tp_vals)
        sex_emb = self.sex(sex_ids)
        stage_emb = self.stage(stage_ids)
        site_emb = self.site(site_ids)

        # For age (feature index 0):
        was_missing_age = 1.0 - cont_mask[:, 0:1] # shape (batch, 1)
        age_emb = age_emb + was_missing_age * self.missing_bias[0]

        # For tumor purity pct (feature index 1):
        was_missing_tumor_purity = 1.0 - cont_mask[:, 1:2] # shape (batch, 1)
        tumor_purity_emb = tumor_purity_emb + was_missing_tumor_purity * self.missing_bias[1]

        clinical_emb = torch.stack([sex_emb, stage_emb, site_emb, age_emb, tumor_purity_emb], dim=1)
        return self.layernorm(clinical_emb)

if __name__ == "__main__":
    df = load_cohort().clinical

    # Smoke test
    means = {
        "age": df["age"].dropna().mean(),
        "tumor_purity_pct": df["tumor_purity_pct"].dropna().mean(),
    }
    cat, cont, mask = ClinicalEmbedding.prepare(df, means)
    print(cat.shape, cat.dtype)
    print(cont.shape, cont.dtype)
    print(mask.shape, mask.dtype)

    model = ClinicalEmbedding()
    out = model(cat, cont, mask)
    print(out.shape)
