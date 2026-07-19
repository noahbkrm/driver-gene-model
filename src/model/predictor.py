import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from data import load_cohort
from constants import HIDDEN_DIM, BATCH
from patient_model import PatientModel
from rna_encoder import RnaStats

class Predictor(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, z_context):
        z_context = self.net(z_context)
        return z_context