import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH, LEARNING_RATE, EMA_PARAM
from patient_model import PatientModel
from rna_encoder import RnaStats, RnaEmbedding
from predictor import Predictor
from dataset_handler import PatientDataset
import copy
from torch.utils.data import DataLoader

def initialize_models(
        rna_stats: RnaStats, 
        n_genes: int,
        hidden_dim: int = HIDDEN_DIM,
        batch_size: int = BATCH,
    ):

    context_model = PatientModel(rna_stats, n_genes, hidden_dim, batch_size)
    target_model = copy.deepcopy(context_model) # Set target to match context 
    
    for param in target_model.parameters():
        param.requires_grad = False
    
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
        ema_param: float = EMA_PARAM,
    ):

    mse_loss_function = nn.MSELoss()
    context_model.train()
    predictor_model.train()
    target_model.eval()

    for batch in loader:

        with torch.no_grad():
            z_target = target_model(batch, mask_snv = False) # Calculate target
        z_context_raw = context_model(batch, mask_snv = True) # Calculate context with masking
        z_context_pred = predictor_model(z_context_raw) # Pass conext to predictor to for stabilization

        optimizer.zero_grad()
        
        # Compute MSE
        loss = mse_loss_function(z_context_pred, z_target)

        loss.backward()
        optimizer.step()
        update_target_model(target_model, context_model, ema_param)

def prepare_dataset(cohort: pd.DataFrame, train_means: dict[str, float], rna_stats: RnaStats):

    dataset = PatientDataset(
        clinical_df=cohort.clinical,
        snv_df=cohort.snv,
        cnv_df=cohort.cnv,
        rna_df=cohort.rna,
        train_means=train_means,
        rna_stats=rna_stats,
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

if __name__ == "__main__":
    train_cohort = load_cohort()

    c_df = train_cohort.clinical
    train_means = {
        "age": c_df["age"].dropna().mean(),
        "tumor_purity_pct": c_df["tumor_purity_pct"].dropna().mean(),
    }
    
    rna_stats = RnaEmbedding.fit(train_cohort.rna)
    
    train_dataset = prepare_dataset(train_cohort, train_means, rna_stats)

    loader = initializeLoader(train_dataset)

    target_model, context_model, predictor, optimizer = initialize_models(
        rna_stats=rna_stats,
        n_genes=rna_stats.n_genes,
    )

    JEPATraining(
        target_model,
        context_model,
        predictor,
        loader,
        optimizer,
    )