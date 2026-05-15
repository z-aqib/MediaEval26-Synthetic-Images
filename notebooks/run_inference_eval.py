#!/usr/bin/env python
# coding: utf-8

# ============================================================
# COMPETITION EVALUATION INFERENCE SCRIPT
# ============================================================
# Runs the top 4 trained models on the competition evaluation dataset
# and saves submission-ready CSV files.
#
# Model paths and thresholds are read automatically from:
#   outputs/best_overall/top4_extraction_log.json  (produced by analyze_experiments.py)
#   outputs/analysis/ranked_experiments.csv         (for thresholds)
#
# Output CSV format per submission file:
#   image_id  - filename without extension
#   prob      - sigmoid probability for synthetic class (1=synthetic)
#   label     - predicted class: 0=real, 1=synthetic
#   threshold - decision threshold used
#
# Generates 4 output files (2 constrained, 2 open).
# ============================================================

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS (derived automatically — no need to edit)
# ============================================================

REPO_ROOT    = Path(__file__).resolve().parent.parent

# ============================================================
# CONFIGURATION — only edit these two lines
# ============================================================

# Path to the competition evaluation dataset folder.
# All images are found recursively; sub-folders are fine.
# After extracting taska_test_R03SaaV7P(1).tar into data/, verify the folder name matches.
EVAL_DATASET_PATH = REPO_ROOT / "data" / "taska_test"

# Team identifier used in output filenames.
TEAM_NAME = "CVG-IBA_MMRG-IBA"
OUTPUT_DIR   = REPO_ROOT / "outputs" / "submission"
LOG_PATH     = REPO_ROOT / "outputs" / "best_overall" / "top4_extraction_log.json"
RANKED_CSV   = REPO_ROOT / "outputs" / "analysis" / "ranked_experiments.csv"
BEST_OVERALL = REPO_ROOT / "outputs" / "best_overall"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# INFERENCE SETTINGS
# ============================================================

BATCH_SIZE  = 32
NUM_WORKERS = 2
IMAGE_SIZE  = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Maps model_folder prefix → model_type key used by MODEL_BUILDERS
MODEL_FOLDER_TO_TYPE = {
    "01_efficientnet_b0": "efficientnet_b0",
    "02_convnext_tiny":   "convnext_tiny",
    "03_clip_vit":        "vit_b16",
}


# ============================================================
# LOAD EXPERIMENTS FROM LOG + RANKED CSV
# ============================================================

def load_experiments() -> list[dict]:
    """
    Build the EXPERIMENTS list automatically from:
      - top4_extraction_log.json  → experiment_id, run_type, model_folder, status, pth_extracted
      - ranked_experiments.csv    → threshold (best-F1 threshold from training)
    """
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Extraction log not found: {LOG_PATH}\n"
            "Run analyze_experiments.py first to generate it."
        )

    with open(LOG_PATH) as f:
        log_entries = json.load(f)

    # Load threshold lookup from ranked_experiments.csv
    threshold_map = {}
    if RANKED_CSV.exists():
        ranked_df = pd.read_csv(RANKED_CSV)
        for _, row in ranked_df.iterrows():
            threshold_map[row["experiment_id"]] = float(row["threshold"])
    else:
        print(f"WARNING: {RANKED_CSV} not found. Thresholds will default to 0.5.")

    # Assign run numbers per run_type (1-indexed, in log order)
    run_counters = defaultdict(int)

    experiments = []
    for entry in log_entries:
        exp_id     = entry["experiment_id"]
        run_type   = entry["run_type"]
        model_fold = entry["model_folder"]
        status     = entry["status"]
        f1_val     = entry.get("f1", None)

        # Resolve model_path based on extraction status
        if status == "extracted_from_git":
            model_path = BEST_OVERALL / f"{model_fold}_{exp_id}" / "best_model.pth"
        elif status == "copied_current_files":
            model_path = BEST_OVERALL / f"current_copy_{model_fold}_{exp_id}" / "best_model.pth"
        else:
            print(f"WARNING: experiment '{exp_id}' has status='{status}' — .pth not available, will skip.")
            model_path = None

        # Resolve model architecture from folder name prefix
        model_type = MODEL_FOLDER_TO_TYPE.get(model_fold)
        if model_type is None:
            print(f"WARNING: unknown model_folder '{model_fold}'. Skipping.")
            continue

        # Threshold from ranked_experiments.csv
        threshold = threshold_map.get(exp_id, 0.5)
        if exp_id not in threshold_map:
            print(f"WARNING: threshold not found for '{exp_id}', defaulting to 0.5.")

        run_counters[run_type] += 1
        run_number = run_counters[run_type]

        experiments.append({
            "run_type":      run_type,
            "run_number":    run_number,
            "experiment_id": exp_id,
            "model_type":    model_type,
            "model_path":    model_path,
            "threshold":     threshold,
            "f1_on_val":     f1_val,
        })

    return experiments


# ============================================================
# MODEL FACTORY
# ============================================================

def create_efficientnet_b0():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def create_convnext_tiny():
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def create_vit_b16():
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 1)
    return model


MODEL_BUILDERS = {
    "efficientnet_b0": create_efficientnet_b0,
    "convnext_tiny":   create_convnext_tiny,
    "vit_b16":         create_vit_b16,
}


def load_model(model_type: str, model_path: Path, device: torch.device) -> nn.Module:
    model = MODEL_BUILDERS[model_type]()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# DATASET — unlabelled evaluation images
# ============================================================

class EvalImageDataset(Dataset):
    """Scans a folder recursively and returns (tensor, image_id, filepath)."""

    def __init__(self, folder: Path, transform):
        self.transform = transform
        self.records   = []

        print(f"Scanning evaluation images in: {folder}")
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                self.records.append({"filepath": p, "image_id": p.stem})

        print(f"Found {len(self.records):,} evaluation images.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec["filepath"]).convert("RGB")
        return self.transform(image), rec["image_id"], str(rec["filepath"])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    all_ids, all_probs = [], []
    with torch.no_grad():
        for tensors, image_ids, _ in tqdm(loader, desc="Inferring", leave=False):
            tensors = tensors.to(device)
            probs   = torch.sigmoid(model(tensors).view(-1)).cpu().numpy().tolist()
            all_ids.extend(list(image_ids))
            all_probs.extend(probs)
    return all_ids, all_probs


def build_submission_df(image_ids, probs, threshold: float) -> pd.DataFrame:
    probs_arr = np.array(probs, dtype=float)
    return pd.DataFrame({
        "image_id":  image_ids,
        "prob":      probs_arr,
        "label":     (probs_arr >= threshold).astype(int),
        "threshold": threshold,
    })


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVAL_DATASET_PATH}\n"
            "Set EVAL_DATASET_PATH at the top of this script."
        )

    print(f"\nLoading experiment configs from: {LOG_PATH}")
    experiments = load_experiments()

    if not experiments:
        raise RuntimeError("No experiments loaded. Check top4_extraction_log.json.")

    print(f"\nLoaded {len(experiments)} experiments:")
    for exp in experiments:
        pth_ok = "✓" if (exp["model_path"] and Path(exp["model_path"]).exists()) else "✗ MISSING"
        print(f"  [{exp['run_type']:>12s} run{exp['run_number']}]  "
              f"threshold={exp['threshold']:.2f}  F1={exp['f1_on_val']}  "
              f"pth={pth_ok}  {exp['experiment_id']}")

    transform   = get_eval_transform()
    eval_ds     = EvalImageDataset(EVAL_DATASET_PATH, transform)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    results_summary = []

    for exp in experiments:
        run_type   = exp["run_type"]
        run_number = exp["run_number"]
        model_path = Path(exp["model_path"]) if exp["model_path"] else None
        threshold  = exp["threshold"]

        print("\n" + "=" * 70)
        print(f"Experiment : {exp['experiment_id']}")
        print(f"Run type   : {run_type}  (run #{run_number})")
        print(f"Model      : {exp['model_type']}")
        print(f"Model path : {model_path}")
        print(f"Threshold  : {threshold}")
        print(f"Val F1     : {exp['f1_on_val']}")
        print("=" * 70)

        if model_path is None or not model_path.exists():
            print(f"SKIPPING — .pth file not found at: {model_path}")
            continue

        print("Loading model...")
        model = load_model(exp["model_type"], model_path, device)

        print("Running inference...")
        image_ids, probs = run_inference(model, eval_loader, device)

        submission_df   = build_submission_df(image_ids, probs, threshold)
        output_filename = f"{TEAM_NAME}_{run_type}_run{run_number}.csv"
        output_path     = OUTPUT_DIR / output_filename
        submission_df.to_csv(output_path, index=False)

        n_synthetic = int((submission_df["label"] == 1).sum())
        n_real      = int((submission_df["label"] == 0).sum())

        print(f"Saved {len(submission_df):,} predictions → {output_path}")
        print(f"  Predicted synthetic : {n_synthetic:,}")
        print(f"  Predicted real      : {n_real:,}")

        results_summary.append({
            "run_type":    run_type,
            "run_number":  run_number,
            "experiment":  exp["experiment_id"],
            "threshold":   threshold,
            "val_f1":      exp["f1_on_val"],
            "n_images":    len(submission_df),
            "n_synthetic": n_synthetic,
            "n_real":      n_real,
            "output_file": str(output_path),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("ALL DONE — SUBMISSION FILES")
    print("=" * 70)
    for r in results_summary:
        print(f"  [{r['run_type']:>12s} run{r['run_number']}]  "
              f"threshold={r['threshold']:.2f}  val_F1={r['val_f1']}  "
              f"→ {Path(r['output_file']).name}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
