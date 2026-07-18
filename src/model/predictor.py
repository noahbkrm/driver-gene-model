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
        self.l1 = nn.Linear(hidden_dim, 1024)
        self.l2 = nn.Linear(1024, hidden_dim)
    
    def forward(self, z_context):
        z_context = self.l1(z_context)
        z_context = F.relu(z_context)
        z_context = self.l2(z_context)
        return z_context