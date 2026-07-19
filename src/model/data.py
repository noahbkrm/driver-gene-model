"""Synthetic 1,000-patient oncology cohort for rare-driver discovery.

All values are integers (includes pandas `pd.NA`).
No file I/O: ``load_cohort()`` regenerates the exact same tables from a seed,
so the dataset can be treated as read-only and reproducible. 

Tables
------
clinical : (1000, 7)    int patient covariates + survival
rna      : (1000, 2000) int negative-binomial-like raw counts
snv      : (1000, 2000) int {0, 1} somatic single-nucleotide variants
cnv      : (1000, 2000) int {-2..2} GISTIC-style copy-number calls

Planted rare driver genes (what the model should recover)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Constants

N_PATIENTS = 5000
N_BACKGROUND = 1990
N_DRIVERS = 10
N_GENES = N_BACKGROUND + N_DRIVERS
N_SITES = 4
CNV_SEGMENT_SIZE = 40
SEED = 20260712

DRIVER_GENES: list[str] = [f"DRV_{i:02d}" for i in range(N_DRIVERS)]
BACKGROUND_GENES: list[str] = [f"GENE_{i:04d}" for i in range(N_BACKGROUND)]
ALL_GENES: list[str] = BACKGROUND_GENES + DRIVER_GENES
PATIENT_IDS: list[str] = [f"PT_{i:04d}" for i in range(N_PATIENTS)]

# Missingness fractions per matrix
MISSING_FRAC = {"rna": 0.003, "snv": 0.010, "cnv": 0.010, "clinical": 0.010}

# Public container
@dataclass
class Cohort:
    clinical: pd.DataFrame
    rna: pd.DataFrame
    snv: pd.DataFrame
    cnv: pd.DataFrame
    driver_genes: list[str]
    responders: dict[str, tuple[np.ndarray, np.ndarray]] # driver -> (gene_idx, log2fc)
    driver_alt: pd.DataFrame

# Builders
def _make_clinical(rng: np.random.Generator, driver_alt: np.ndarray) -> pd.DataFrame:
    age = rng.integers(20, 86, N_PATIENTS)
    sex = rng.integers(0, 2, N_PATIENTS)
    stage = rng.choice([1,2,3,4], size = N_PATIENTS, p = [0.20, 0.30, 0.35, 0.15])
    purity = rng.integers(30, 96, N_PATIENTS)
    site = rng.integers(0, N_SITES, N_PATIENTS)

    driver_effects = np.array([
        0.25,
        0.10,
        0.04,
        0.20,
        0.0,
        0.10,
        0.0,
        0.24,
        -0.04,
        0.0
    ])

    hazard = (
        0.05 * stage
        + driver_alt @ driver_effects
    )

    # true survival time
    true_os = rng.exponential(scale=1 / hazard)

    # administrative censoring
    follow_up = rng.uniform(low=24, high=120, size=N_PATIENTS)

    # observed survival
    os_months = np.minimum(true_os,follow_up).astype(np.float32)

    # event occurred before censoring
    os_event = (true_os <= follow_up).astype(np.int8)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "stage": stage,
            "tumor_purity_pct": purity,
            "site": site,
            "os_months": os_months,
            "os_event": os_event,
        },
        index=pd.Index(PATIENT_IDS, name="patient_id"),
    )

    # Integer clinical variables
    for col in [
        "age",
        "sex",
        "stage",
        "tumor_purity_pct",
        "site",
        "os_event",
    ]:
        df[col] = df[col].astype("Int32")

    # Keep survival time continuous
    df["os_months"] = df["os_months"].astype("float32")

    return df

def _pick_responders(
        rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each driver, pick ~20 background genes it purturbs, with log2 FCs."""
    responders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for drv in DRIVER_GENES:
        k = int(rng.integers(15,26))
        idx = rng.choice(N_BACKGROUND, size = k, replace = False)
        lfc = rng.choice([-1.5, -1.0, 1.0, 1.5, 2.0], size = k)
        responders[drv] = (idx, lfc.astype(np.float32))
    return responders

def _make_driver_alterations(rng: np.random.Generator) -> np.ndarray:
    """Return a boolean mask of altered drivers"""
    prevalence = rng.uniform(0.05, 0.09, N_DRIVERS)
    alt = rng.random((N_PATIENTS, N_DRIVERS)) < prevalence
    # Gurantee 8 hits per driver
    for j in range(N_DRIVERS):
        need = 8 - int(alt[:, j].sum())
        if need > 8:
            extra = rng.choice(np.where(~alt[:, j])[0], size = need, replace = False)
            alt[extra, j] = True
    return alt

def _make_rna(
    rng: np.random.Generator, 
    site: np.ndarray,
    driver_alt: np.ndarray,
    responders: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    gene_mean = np.exp(rng.normal(3.0, 2.0, N_GENES)).clip(0.5, 5000.0)
    # 30% of genes are silent
    silent_mask = rng.random(N_GENES) < 0.30
    protected = np.zeros(N_GENES, dtype = bool)
    protected[N_BACKGROUND:] = True # driver columns
    for idx, _ in responders.values():
        protected[idx] = True
    silent_mask &= ~protected
    gene_mean[silent_mask] = rng.uniform(0.001, 0.03, silent_mask.sum())
    # Per-patient library-size scaling factor
    lib = rng.gamma(shape = 20.0, scale = 1.0 / 20.0, size = N_PATIENTS)
    # Per-site library size shift + a per-fene batch effect on 5% of genes
    site_lib = np.array([0.85, 1.0, 1.10, 1.20])[site]
    batch_effect = np.ones((N_SITES, N_GENES), dtype = np.float32)
    n_batch_genes = int(0.05 * N_GENES)
    for s in range(N_SITES):
        g = rng.choice(N_GENES, size = n_batch_genes, replace = False)
        batch_effect[s, g] = np.exp(rng.normal(0.0, 0.3, n_batch_genes)).astype(np.float32)
    
    # Mean matrix (patients x genes) before driver purturbation
    mu = (lib * site_lib)[:, None] * gene_mean[None, :] * batch_effect[site]

    # Apply driver responder programs (only where the driver is altered)
    for j, drv in enumerate(DRIVER_GENES):
        idx, lfc = responders[drv]
        hit = driver_alt[:, j]
        if hit.any():
            mu[np.ix_(hit, idx)] *= (2.0 ** lfc)[None, :]

    # Negative binomial draw around mu with a fixed dispersion (r=5)
    r = 5.0
    p = r / (r + mu)
    counts = rng.negative_binomial(r, p).astype(np.int32)

    return pd.DataFrame(counts, index = PATIENT_IDS, columns = ALL_GENES).astype("Int32")

def _make_snv(rng: np.random.Generator, driver_alt: np.ndarray) -> pd.DataFrame:
    # Background per-gene mutation rate
    base_rate = rng.beta(0.5, 200.0, N_GENES).astype(np.float32)
    # Per-patient burden multiplier
    burden = rng.gamma(2.0, 0.5, N_PATIENTS).astype(np.float32)
    burden[rng.random(N_PATIENTS) < 0.03] *= 8.0

    prob = np.clip(burden[:, None] * base_rate[None, :], 0.0, 0.9)
    snv = (rng.random((N_PATIENTS, N_GENES)) < prob).astype(np.int8)

    # About half of each driver's alterations come through SNV
    for j, drv in enumerate(DRIVER_GENES):
        col = N_BACKGROUND + j
        hit = driver_alt[:, j]
        snv_hit = hit & (rng.random(N_PATIENTS) < 0.6)
        snv[snv_hit, col] = 1
    
    return pd.DataFrame(snv, index = PATIENT_IDS, columns = ALL_GENES).astype("Int8")

def _make_cnv(rng: np.random.Generator, driver_alt: np.ndarray) -> pd.DataFrame:
    cnv = np.zeros((N_PATIENTS, N_GENES), dtype = np.int8)

    # Segments gains/losses for each patient
    n_segments = N_GENES // CNV_SEGMENT_SIZE
    for p in range(N_PATIENTS):
        n_events = rng.poisson(3.0)
        if n_events == 0:
            continue
        segs = rng.integers(0, n_segments, size = n_events)
        levels = rng.choice([-1, 1], size = n_events, p = [0.5, 0.5])
        levels[rng.random(n_events) < 0.15] *= 2
        for seg, lvl in zip(segs, levels):
            start = seg * CNV_SEGMENT_SIZE
            end = start + CNV_SEGMENT_SIZE
            cnv[p, start:end] = lvl
    
    # Drivers not hit by SNV path get a high-level focal CNA at their column
    for j, drv in enumerate(DRIVER_GENES):
        col = N_BACKGROUND + j
        hit = driver_alt[:, j] & (cnv[:, col] == 0)
        cna_hit = hit & (rng.random(N_PATIENTS) < 0.7)
        # Half amplifications, half deletions
        signs = rng.choice([-2, 2], size = int(cna_hit.sum()))
        cnv[np.where(cna_hit)[0], col] = signs
    
    return pd.DataFrame(cnv, index = PATIENT_IDS, columns = ALL_GENES).astype("Int8")

# Missingness
def _inject_missing(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    if frac <= 0:
        return df
    n_rows, n_cols = df.shape
    n_missing = int(frac * n_rows * n_cols)
    rows = rng.integers(0, n_rows, n_missing)
    cols = rng.integers(0, n_cols, n_missing)
    out = df.copy()
    for r, c in zip(rows, cols):
        out.iat[r, c] = pd.NA
    return out

# Public entry point
def load_cohort(seed: int = SEED) -> Cohort:
    """Regngerate the full cohort determnistically"""
    rng = np.random.default_rng(seed)

    driver_alt = _make_driver_alterations(rng)
    responders = _pick_responders(rng)

    # Clinical needs driver_alt for the survival prior, site drives RNA batch effect
    clinical = _make_clinical(rng, driver_alt)
    site = clinical["site"].to_numpy(dtype = np.int64)

    rna = _make_rna(rng, site, driver_alt, responders)
    snv = _make_snv(rng, driver_alt)
    cnv = _make_cnv(rng, driver_alt)

    rna = _inject_missing(rna, MISSING_FRAC["rna"], rng)
    snv = _inject_missing(snv, MISSING_FRAC["snv"], rng)
    cnv = _inject_missing(cnv, MISSING_FRAC["cnv"], rng)
    clinical = _inject_missing(clinical, MISSING_FRAC["clinical"], rng)

    driver_alt_df = pd.DataFrame(
        driver_alt.astype("int8"),
        index=PATIENT_IDS,
        columns=DRIVER_GENES
    )

    return Cohort(
        clinical = clinical,
        rna = rna,
        snv = snv,
        cnv = cnv,
        driver_genes = DRIVER_GENES,
        responders = responders,
        driver_alt=driver_alt_df
    )

def main() -> None:
    c = load_cohort()
    print(f"clinical : {c.clinical.shape}")
    print(f"rna : {c.rna.shape}")
    print(f"snv : {c.snv.shape}")
    print(f"cnv : {c.cnv.shape}")
    print()

if __name__ == "__main__":
    main()