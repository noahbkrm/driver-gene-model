import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH, LEARNING_RATE, EMA_PARAM
from patient_model import PatientModel
from rna_encoder import RnaStats
from predictor import Predictor
from dataset_handler import PatientDataset
import copy
from torch.utils.data import DataLoader

def InitializeModels(
        rna_stats: RnaStats, 
        n_genes: int,
        hidden_dim: int = HIDDEN_DIM,
        batch_size: int = BATCH,
    ):

    target_model = PatientModel(rna_stats, n_genes, hidden_dim, batch_size)
    context_model = PatientModel(rna_stats, n_genes, hidden_dim, batch_size)
    target_model.load_state_dict(copy.deepcopy(context_model.state_dict())) # Set target weights to match context weights
    predictor_model = Predictor(hidden_dim) 

    optimizer = torch.optim.AdamW(
        list(context_model.parameters()) +
        list(predictor_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-2,
    )

    return target_model, context_model, predictor_model, optimizer

def update_target_model(
    target_model: nn.Module,
    context_model: nn.Module,
    ema_param: float = EMA_PARAM,
    ):
    with torch.no_grad():
        for target_param, context_param in zip(
            target_model.parameters(),
            context_model.parameters()
        ):
            target_param.data = (  #thetaT = thetaT*m + (1-m)thetaC
                ema_param * target_param.data
                +
                (1 - ema_param) * context_param.data
            )

def JEPATraining(
        target_model: PatientModel,
        context_model: PatientModel,
        predictor_model: Predictor,
        loader: DataLoader,
        optimizer,
        ema_param: int = EMA_PARAM,
    ):

    for batch in loader:

        with torch.no_grad():
            z_target = target_model(batch, mask_snv = False) # Calculate target
        z_context_raw = context_model(batch, mask_snv = True) # Calculate context with masking
        z_context_pred = predictor_model(z_context_raw) # Pass conext to predictor to for stabilization

        # Compute MSE
        mse_loss_function = nn.MSELoss()
        loss = mse_loss_function(z_context_pred, z_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target_model(target_model, context_model, ema_param)

def prepareDataset():
    cohort = load_cohort()

    c_df = cohort.clinical
    train_means = {
        "age": c_df["age"].dropna().mean(),
        "tumor_purity_pct": c_df["tumor_purity_pct"].dropna().mean(),
    }

    dataset = PatientDataset(
        clinical_df=cohort.clinical,
        snv_df=cohort.snv,
        cnv_df=cohort.cnv,
        rna_df=cohort.rna,
        train_means=train_means,
        rna_stats=RnaStats,
    )

    return dataset

def initializeLoader(dataset: PatientDataset, batch_size: int = BATCH):
    # Data loading
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader