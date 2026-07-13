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

