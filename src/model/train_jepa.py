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
        scaler,
        ema_param: float = EMA_PARAM,
    ):

    mse_loss_function = nn.MSELoss()
    context_model.train()
    predictor_model.train()
    target_model.eval()
    total_loss = 0
    for batch in loader:

        optimizer.zero_grad()

        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                z_target = target_model(batch, mask_snv = False) # Calculate target

            z_context_raw = context_model(batch, mask_snv = True) # Calculate context with masking
            z_context_pred = predictor_model(z_context_raw) # Pass conext to predictor to for stabilization

            # Compute MSE
            loss = mse_loss_function(z_context_pred, z_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_target_model(target_model, context_model, ema_param)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(
        "Embedding std:",
        z_context_pred.std().item()
    )
    cos = F.cosine_similarity(
        z_context_pred,
        z_target,
        dim=-1
    ).mean()

    print(f"Cosine similarity: {cos.item():.4f}")

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

def generate_embeddings(model, loader, device):

    model.eval()

    embeddings = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                k:v.to(device)
                for k,v in batch.items()
            }

            z = model(batch, mask_snv=False)

            embeddings.append(
                z.cpu()
            )

    return torch.cat(embeddings, dim=0)

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

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(NUM_EPOCHS):
        loss = JEPATraining(
            target_model,
            context_model,
            predictor,
            loader,
            optimizer,
            scaler,
        ) 
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}") 
    
    torch.save(  # Save JEPA checkpoint for reference later if needed
        {
            "context_model": context_model.state_dict(),
            "target_model": target_model.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        "jepa_checkpoint.pt"
    )

    context_model.eval() # Switch context model into eval mode

    for p in context_model.parameters(): # Freeze the context model
        p.requires_grad = False
            
    embeddings = generate_embeddings( # Generate the patient-level embeddings
        context_model,
        loader,
        device
    )
    print(embeddings.shape)

    torch.save(embeddings, "patient_embeddings.pt")

    embedding_df = pd.DataFrame(embeddings.numpy())
    embedding_df["patient_id"] = train_cohort.clinical.index
    embedding_df.to_csv("patient_embeddings.csv", index=False)