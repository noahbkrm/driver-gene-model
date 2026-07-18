import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH, LEARNING_RATE, EMA_PARAM
from patient_model import PatientModel
from rna_encoder import RnaStats
from predictor import Predictor
import copy

def JEPATraining(
        rna_stats: RnaStats, 
        n_genes: int,
        batch: dict[str, pd.DataFrame],
        hidden_dim: int = HIDDEN_DIM,
        batch_size: int = BATCH,
    ):

    # Initialize models
    target_model = PatientModel(rna_stats, n_genes, hidden_dim, batch_size)
    context_model = PatientModel(rna_stats, n_genes, hidden_dim, batch_size)

    target_model.load_state_dict(copy.deepcopy(context_model.state_dict())) # Set target weights to match context weights

    predictor_model = Predictor(hidden_dim) 

    z_target = target_model(batch, mask_snv = False) # Calculate target
    z_context_raw = context_model(batch, mask_snv = True) # Calculate context with masking
    z_context_pred = predictor_model(z_context_raw) # Pass conext to predictor to for stabilization

    # Compute MSE
    mse_loss_function = nn.MSELoss()
    loss = mse_loss_function(z_context_pred, z_target)
