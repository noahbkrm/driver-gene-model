import torch
from torch.utils.data import Dataset
import pandas as pd
from rna_encoder import RnaStats, RnaEmbedding
from snv_encoder import SNVEmbedding
from cnv_encoder import CNVEmbedding
from clinical_encoder import ClinicalEmbedding

class PatientDataset(Dataset):

    def __init__(
            self, 
            clinical_df: pd.DataFrame,
            snv_df: pd.DataFrame,
            cnv_df: pd.DataFrame,
            rna_df: pd.DataFrame,
            train_means: dict[str,float],
            rna_stats: RnaStats,
        ):
        super().__init__()
        clinical_cat, clinical_cont, clinical_mask = ClinicalEmbedding.prepare(clinical_df, train_means)
        rna_expression, rna_mask = RnaEmbedding.prepare(rna_df, rna_stats)
        cnv_states = CNVEmbedding.prepare(cnv_df)
        snv_states = SNVEmbedding.prepare(snv_df)

        self.clinical_cat = clinical_cat
        self.clinical_cont = clinical_cont
        self.clinical_mask = clinical_mask

        self.rna_expression = rna_expression
        self.rna_mask = rna_mask

        self.cnv_states = cnv_states
        self.snv_states = snv_states

        # Check number of patients
        self.n_patients = self.tensors["clinical_cat"].shape[0]

        assert self.tensors["clinical_cat"].shape[0] == self.tensors["rna_expression"].shape[0]
        assert self.tensors["clinical_cat"].shape[0] == self.tensors["cnv_states"].shape[0]
        assert self.tensors["clinical_cat"].shape[0] == self.tensors["snv_states"].shape[0]

    def __len__(self):
        return self.n_patients

    def __getitem__(self, idx):

        patient = {
            "clinical_cat": self.clinical_cat["clinical_cat"][idx],
            "clinical_cont": self.clinical_cont["clinical_cont"][idx],
            "clinical_mask": self.clinical_mask["clinical_mask"][idx],

            "rna_expression": self.rna_expression["rna_expression"][idx],
            "rna_mask": self.rna_mask["rna_mask"][idx],

            "cnv_states": self.cnv_states["cnv_states"][idx],

            "snv_states": self.snv_states["snv_states"][idx],
        }

        return patient