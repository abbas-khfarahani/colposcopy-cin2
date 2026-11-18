"""Locked protocol constants. Change here, not in scattered scripts."""

from __future__ import annotations

from pathlib import Path

from ml.utils.paths import ML_ROOT, PROJECT_ROOT, default_my_dataset_root, default_splits_dir

PROTOCOL_VERSION = "1.0.0"
PROTOCOL_NAME = "patient_cv_v1"

RUNS_DIR = ML_ROOT / "runs" / "protocol"
LEGACY_SPLITS_DIR = default_splits_dir()
DATASET_ROOT = default_my_dataset_root()

# --- evaluation ---
N_OUTER_FOLDS = 5
N_INNER_VAL_FOLDS = 5  # 1/5 of outer-train used as inner validation
MIN_POSITIVES_PER_FOLD = 8
SCREENING_SPLIT_SEEDS: tuple[int, ...] = (42,)
CONFIRM_SPLIT_SEEDS: tuple[int, ...] = (42, 73, 101)
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42

# --- optimization (design-doc anchors) ---
IMAGE_SIZE = 224
MAX_EPOCHS = 40
MIN_EPOCHS = 8
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
BATCH_SIZE = 32
MIL_BATCH_PATIENTS = 8
BACKBONE_LR_MULT = 1.0
DROPOUT = 0.25
ATTN_HIDDEN = 256
MIL_EMBED_DIM = 512
MIL_ATTN_HIDDEN = 128
MIL_DROPOUT = 0.25
CENTER_SIGMA = 0.45
AMP = True

# --- SSL ---
SSL_EPOCHS = 100
SSL_LR = 3e-4
SSL_BATCH_SIZE = 32
SSL_TEMPERATURE = 0.2
SSL_PROJECTOR_DIM = 128

# --- operating points ---
OP_SENS_PRIMARY = 0.70
OP_SENS_RESEARCH = 0.80
STACK_C_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)
# Stronger / weaker C for image+clinical fusion (B11). Does not change B0.
FUSION_C_GRID: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
STACK_IMAGE_CANDIDATES: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B6")

# Architecture search is frozen. Do not add new CNN/MIL/SSL IDs.
# Nested CV estimates the selected-model OOF of this shortlist only.
FROZEN_NESTED_CANDIDATES: tuple[str, ...] = ("B0", "B1", "B4", "B11")
LSE_TAUS: tuple[float, ...] = (0.25, 0.5, 1.0)

# --- clinical one-hot (must match ml.data.clinical) ---
HPV_BUCKETS = ("neg", "hr16_18", "other_hr", "missing")
CYTO_BUCKETS = ("nilm", "ascus", "lsil", "asch", "hsil_plus", "other", "missing")
CLINICAL_FEATURE_NAMES: tuple[str, ...] = (
    "age_z",
    "age_missing",
    *(f"hpv_{b}" for b in HPV_BUCKETS),
    *(f"cyto_{b}" for b in CYTO_BUCKETS),
)

HEAD_PARAM_KEYS = (
    "fc",
    "classifier",
    "head",
    "attn",
    "proj",
    "mil",
    "projector",
)


def experiment_dir(root: Path, experiment_id: str, split_seed: int) -> Path:
    return root / "experiments" / experiment_id / f"seed{split_seed}"
