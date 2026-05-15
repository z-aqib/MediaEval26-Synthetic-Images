#!/usr/bin/env python
# coding: utf-8

# ============================================================
# COMPETITION EVALUATION INFERENCE SCRIPT
# ============================================================
# Runs the top 4 trained models on the competition evaluation dataset
# and saves submission-ready CSV files.
#
# Output CSV format per submission file:
#   image_id  - filename without extension
#   prob      - sigmoid probability for synthetic class (1=synthetic)
#   label     - predicted class: 0=real, 1=synthetic
#   threshold - decision threshold used
#
# Generates 4 output files (2 constrained, 2 open).
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ConvNeXt_Tiny_Weights,
    ViT_B_16_Weights,
)
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION — EDIT THIS SECTION BEFORE RUNNING
# ============================================================

# Path to the competition evaluation dataset folder.
# Can contain sub-folders; all images are found recursively.
EVAL_DATASET_PATH = Path(r"PATH_TO_EVAL_DATASET")

# Output folder for submission CSV files.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "submission"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Team identifier used in output filenames.
TEAM_NAME = "CVG-IBA_MMRG-IBA"

# Batch size for inference (increase if you have more GPU RAM).
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 224

# ImageNet normalization — same as used during training.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Supported image extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================
# TOP 4 EXPERIMENT CONFIGURATIONS
# ============================================================
# Fill in the .pth file paths for each of the 4 experiments.
# Thresholds come from the best-F1 threshold found during training
# (stored in ranked_experiments.csv).
#
# model_type options: "efficientnet_b0" | "convnext_tiny" | "vit_b16"
# ============================================================

EXPERIMENTS = [
    {
        "run_type": "constrained",
        "run_number": 1,
        "experiment_id": "20260514_105614_zuha_01_efficientnet_b0_constrained_ep5_bs32_lr0p0001_basic",
        "model_type": "efficientnet_b0",
        "model_path": REPO_ROOT / "outputs" / "best_overall"
                      / "01_efficientnet_b0_20260514_105614_zuha_01_efficientnet_b0_constrained_ep5_bs32_lr0p0001_basic"
                      / "best_model.pth",
        "threshold": 0.65,
        "f1_on_val": 0.7403,
    },
    {
        "run_type": "constrained",
        "run_number": 2,
        "experiment_id": "20260514_130821_zuha_01_efficientnet_b0_constrained_ep10_bs32_lr0p0001_basic",
        "model_type": "efficientnet_b0",
        # Update this path to where you have the .pth for this experiment.
        "model_path": REPO_ROOT / "outputs" / "best_overall"
                      / "01_efficientnet_b0_20260514_130821_zuha_01_efficientnet_b0_constrained_ep10_bs32_lr0p0001_basic"
                      / "best_model.pth",
        "threshold": 0.65,
        "f1_on_val": 0.7403,
    },
    {
        "run_type": "open",
        "run_number": 1,
        "experiment_id": "20260514_205725_zuha_01_efficientnet_b0_open_ep5_bs32_lr0p0001_basic",
        "model_type": "efficientnet_b0",
        "model_path": REPO_ROOT / "outputs" / "best_overall"
                      / "01_efficientnet_b0_20260514_205725_zuha_01_efficientnet_b0_open_ep5_bs32_lr0p0001_basic"
                      / "best_model.pth",
        "threshold": 0.45,
        "f1_on_val": 0.71875,
    },
    {
        "run_type": "open",
        "run_number": 2,
        "experiment_id": "20260514_194804_zuha_01_efficientnet_b0_open_ep8_bs32_lr5em05_basic",
        "model_type": "efficientnet_b0",
        "model_path": REPO_ROOT / "outputs" / "best_overall"
                      / "01_efficientnet_b0_20260514_194804_zuha_01_efficientnet_b0_open_ep8_bs32_lr5em05_basic"
                      / "best_model.pth",
        "threshold": 0.40,
        "f1_on_val": 0.7165,
    },
]


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
    # ConvNeXt-Tiny classifier: model.classifier[-1] is a Linear layer.
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
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: {list(MODEL_BUILDERS.keys())}")

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
        self.records = []

        print(f"Scanning evaluation images in: {folder}")
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                # image_id is the filename stem (no extension), matching submission format.
                self.records.append({"filepath": p, "image_id": p.stem})

        print(f"Found {len(self.records):,} evaluation images.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["filepath"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, rec["image_id"], str(rec["filepath"])


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
    """Return parallel lists: image_ids, probs."""
    all_ids   = []
    all_probs = []

    with torch.no_grad():
        for tensors, image_ids, _ in tqdm(loader, desc="Inferring", leave=False):
            tensors = tensors.to(device)
            logits  = model(tensors).view(-1)
            probs   = torch.sigmoid(logits).cpu().numpy().tolist()

            all_ids.extend(list(image_ids))
            all_probs.extend(probs)

    return all_ids, all_probs


def build_submission_df(image_ids, probs, threshold: float) -> pd.DataFrame:
    probs_arr  = np.array(probs, dtype=float)
    labels_arr = (probs_arr >= threshold).astype(int)

    return pd.DataFrame({
        "image_id":  image_ids,
        "prob":      probs_arr,
        "label":     labels_arr,
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
            "Update EVAL_DATASET_PATH at the top of this script."
        )

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

    for exp in EXPERIMENTS:
        run_type   = exp["run_type"]
        run_number = exp["run_number"]
        model_path = Path(exp["model_path"])
        threshold  = exp["threshold"]

        print("\n" + "=" * 70)
        print(f"Experiment : {exp['experiment_id']}")
        print(f"Run type   : {run_type}  (run #{run_number})")
        print(f"Model      : {exp['model_type']}")
        print(f"Model path : {model_path}")
        print(f"Threshold  : {threshold}")
        print(f"Val F1     : {exp['f1_on_val']}")
        print("=" * 70)

        if not model_path.exists():
            print(f"WARNING: Model file not found at {model_path} — skipping.")
            print("Update the model_path in the EXPERIMENTS list at the top of this script.")
            continue

        print("Loading model...")
        model = load_model(exp["model_type"], model_path, device)

        print("Running inference...")
        image_ids, probs = run_inference(model, eval_loader, device)

        submission_df = build_submission_df(image_ids, probs, threshold)

        # Output filename: TEAM_constrained_run1.csv  /  TEAM_open_run1.csv
        output_filename = f"{TEAM_NAME}_{run_type}_run{run_number}.csv"
        output_path     = OUTPUT_DIR / output_filename
        submission_df.to_csv(output_path, index=False)

        n_synthetic = int((submission_df["label"] == 1).sum())
        n_real      = int((submission_df["label"] == 0).sum())

        print(f"Saved {len(submission_df):,} predictions to: {output_path}")
        print(f"  Predicted synthetic : {n_synthetic:,}")
        print(f"  Predicted real      : {n_real:,}")

        results_summary.append({
            "run_type":   run_type,
            "run_number": run_number,
            "experiment": exp["experiment_id"],
            "threshold":  threshold,
            "val_f1":     exp["f1_on_val"],
            "n_images":   len(submission_df),
            "n_synthetic": n_synthetic,
            "n_real":     n_real,
            "output_file": str(output_path),
        })

        # Free GPU memory before loading next model.
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\n" + "=" * 70)
    print("ALL DONE — SUBMISSION FILES SUMMARY")
    print("=" * 70)
    for r in results_summary:
        print(f"[{r['run_type']:>12s} run{r['run_number']}]  "
              f"threshold={r['threshold']:.2f}  "
              f"val_F1={r['val_f1']:.4f}  "
              f"→ {Path(r['output_file']).name}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
