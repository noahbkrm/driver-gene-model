import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import load_cohort
from constants import *
from patient_model import PatientModel
from rna_encoder import RnaStats, RnaEmbedding
from predictor import Predictor
from dataset_handler import PatientDataset
import copy
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models(
        rna_stats: RnaStats, 
        n_genes: int,
        n_variant_genes: int,
        hidden_dim: int = HIDDEN_DIM,
        batch_size: int = BATCH,
    ):

    context_model = PatientModel(rna_stats, n_genes, n_variant_genes, hidden_dim, batch_size)
    target_model = copy.deepcopy(context_model) # Set target to match context 
    
    for param in target_model.parameters():
        param.requires_grad = False
    
    predictor_model = Predictor(hidden_dim) 

    # Move models to GPU
    target_model.to(device)
    context_model.to(device)
    predictor_model.to(device)

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
    total_loss = 0
    for batch in loader:

        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

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
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(
        "Embedding std:",
        z_context_pred.std().item()
    )
    return avg_loss

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
        shuffle=True,
        pin_memory=True,
    )

    return loader


def smoke_test(
        target_model,
        context_model,
        predictor_model,
        loader,
        optimizer,
    ):

    print("Running smoke test...")

    # Get one batch
    batch = next(iter(loader))

    print("\nOriginal batch:")
    for key, value in batch.items():
        print(
            key,
            value.shape,
            value.dtype,
            value.device
        )

    # Move batch to GPU
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    print("\nMoved batch:")
    for key, value in batch.items():
        print(
            key,
            value.shape,
            value.dtype,
            value.device
        )

    # Forward pass
    target_model.eval()
    context_model.train()
    predictor_model.train()

    with torch.no_grad():
        z_target = target_model(
            batch,
            mask_snv=False
        )

    z_context = context_model(
        batch,
        mask_snv=True
    )

    z_pred = predictor_model(z_context)

    print("\nEmbeddings:")
    print("z_target:", z_target.shape)
    print("z_context:", z_context.shape)
    print("z_pred:", z_pred.shape)

    # Loss
    loss_fn = nn.MSELoss()

    loss = loss_fn(
        z_pred,
        z_target
    )

    print("\nLoss:")
    print(loss)

    # Backprop
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("\nBackprop successful!")

    # Check GPU memory
    if torch.cuda.is_available():
        print("\nGPU memory:")
        print(
            torch.cuda.memory_allocated()/1024**2,
            "MB allocated"
        )

    print("\nSmoke test passed!")

if __name__ == "__main__":
    train_cohort = load_cohort()

    n_variant_genes = train_cohort.cnv.shape[1]
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
        n_variant_genes=n_variant_genes
    )

    for epoch in range(NUM_EPOCHS):
        loss = JEPATraining(
            target_model,
            context_model,
            predictor,
            loader,
            optimizer,
        ) 
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}")
        