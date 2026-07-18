import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH
from patient_model import PatientModel
from rna_encoder import RnaStats

def JEPATraining(
        model: PatientModel,
        rna_stats: RnaStats, 
        n_genes: int,
        data: dict[str, pd.DataFrame],
        hidden_dim: int = HIDDEN_DIM,
        batch_size: int = BATCH,
    ):

    target_model = model(rna_stats, n_genes, hidden_dim, batch_size)
    context_model = model(rna_stats, n_genes, hidden_dim, batch_size)

    