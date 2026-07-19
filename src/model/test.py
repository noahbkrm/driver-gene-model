import torch
import torch.nn as nn
import pandas as pd
from data import load_cohort
from constants import *
from patient_model import PatientModel
from rna_encoder import RnaStats, RnaEmbedding
from predictor import Predictor
from dataset_handler import PatientDataset
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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