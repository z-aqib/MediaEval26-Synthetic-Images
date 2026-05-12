#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


# ============================================================
# 3. IMPORTS
# ============================================================
# These imports are shared across most model notebooks.

import os
import json
import random
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

import torch

print("Imports completed successfully.")

# ============================================================
# 15. CONVNEXT-TINY IMPORTS
# ============================================================
# These imports are specific to this model notebook.

from tqdm.auto import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

print("ConvNeXt-Tiny-specific imports completed.")


# In[ ]:


# ============================================================
# 4. SEED AND DEVICE SETUP
# ============================================================
# Setting seeds helps make experiments more reproducible.
# Some GPU operations may still have small randomness, but this reduces variation.

def set_seed(seed):
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value used for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes PyTorch more deterministic.
    # This can sometimes make training slightly slower, but results become more stable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Reproducibility
SEED = 42

set_seed(SEED)

# Use GPU if Kaggle gives CUDA access, otherwise use CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Seed set to: {SEED}")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU detected. Make sure Kaggle accelerator is turned on if training.")


# # Variables
# CHANGE THIS ONLYYY

# In[ ]:


# ============================================================
# 1. SHARED PARAMETER BLOCK
# ============================================================
# Change these values before each experiment run.
# All important experiment settings should stay here so nobody has to search inside the code.

RUNNER_NAME = "zuha"                       # Change to: "zuha", "izma", or "fatima"
MODEL_NAME = "convnext_tiny"               # This notebook is for ConvNeXt-Tiny experiments
RUN_TYPE = "constrained"                   # Use "constrained" or "open"

# General training settings
EPOCHS = 5
BATCH_SIZE = 24                            # ConvNeXt is heavier than EfficientNet, so 24 is safer on Kaggle
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"                   # Options: "adam", "adamw", "sgd"
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "none"                    # Options: "none", "cosine", "step"
LOSS_FUNCTION_NAME = "BCEWithLogitsLoss"

# Image/model settings
IMAGE_SIZE = 224
PRETRAINED = True
FREEZE_BACKBONE = False
AUGMENTATION_TYPE = "basic"                # Options: "basic", "light_aug", "jpeg_like"

# Reproducibility
SEED = 42

# Binary classification settings
# Label convention:
# 0 = real
# 1 = synthetic
THRESHOLD = 0.5

# Notes for this experiment.
EXPERIMENT_NOTES = "ConvNeXt-Tiny baseline with basic augmentation"

print("Shared parameters loaded.")
print(f"Runner: {RUNNER_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"Run type: {RUN_TYPE}")

# For your first test run, make it lighter:

# EPOCHS = 2
# MAX_TRAIN_IMAGES = 3000
# MAX_EVAL_IMAGES = 1000

# For the real baseline later:

# EPOCHS = 5
# MAX_TRAIN_IMAGES = None
# MAX_EVAL_IMAGES = None

# ============================================================
# 16. CONVNEXT-TINY EXTRA PARAMETERS
# ============================================================
# These are extra settings for the actual training loop.

NUM_WORKERS = 2                  # Kaggle usually works well with 2
PIN_MEMORY = True                # Speeds up GPU loading if CUDA is available
USE_AMP = True                   # Mixed precision training, faster on GPU
GRAD_CLIP_NORM = None            # Example: set 1.0 if gradients become unstable

# Best model selection.
# For this competition, F1 matters most, so we save the model with best validation F1.
SAVE_BEST_BY = "f1"              # Options: "f1" or "val_loss"

# To avoid training/evaluation leakage, remove eval image paths from train_df.
REMOVE_EVAL_FROM_TRAIN = True

print("ConvNeXt-Tiny extra parameters loaded.")
print(f"NUM_WORKERS: {NUM_WORKERS}")
print(f"USE_AMP: {USE_AMP}")
print(f"SAVE_BEST_BY: {SAVE_BEST_BY}")


# In[ ]:


# ============================================================
# 2. DATASET SELECTION BLOCK
# ============================================================
# Training datasets can be changed by the runner.
# Evaluation dataset should remain the same for ALL models for fair comparison.

# ----------------------------
# Training dataset choices
# ----------------------------
USE_WANG_TRAIN = True
USE_CORVI_TRAIN = True

# Keep this False if DMImageDetect train is also being used as evaluation.
# We do not want training/evaluation leakage.
USE_DMIMAGEDETECT_TRAIN = False

USE_REALRAISE_TRAIN = False

# ----------------------------
# Fixed evaluation dataset
# ----------------------------
# IMPORTANT:
# Keep this same across all model notebooks unless the whole team agrees to change it.
EVALUATION_DATASET_KEY = "dmimagedetect_train"

# Limit images for quick debugging.
# Use small numbers first to check if the notebook works.
# Then set both to None for full run.
MAX_TRAIN_IMAGES = 3000
MAX_EVAL_IMAGES = 1000

# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

print("Dataset selection loaded.")
print("Training dataset choices:")
print(f"  Wang train: {USE_WANG_TRAIN}")
print(f"  Corvi train: {USE_CORVI_TRAIN}")
print(f"  DMImageDetect train: {USE_DMIMAGEDETECT_TRAIN}")
print(f"  RealRAISE train: {USE_REALRAISE_TRAIN}")
print(f"Fixed evaluation dataset key: {EVALUATION_DATASET_KEY}")


# In[ ]:


# ============================================================
# 5. OUTPUT PATHS AND EXPERIMENT ID
# ============================================================
# We keep normal filenames for latest outputs.
# The summary CSV appends every run, while other files are overwritten with latest run.

# Timestamp makes each row in the summary CSV uniquely traceable.
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Clean learning rate text for experiment ID.
lr_text = str(LEARNING_RATE).replace(".", "p").replace("-", "m")

# Experiment ID goes inside the summary CSV.
# We are not using it for filenames because we want simple latest-output filenames.
EXPERIMENT_ID = (
    f"{TIMESTAMP}_{RUNNER_NAME}_{MODEL_NAME}_{RUN_TYPE}"
    f"_ep{EPOCHS}_bs{BATCH_SIZE}_lr{lr_text}_{AUGMENTATION_TYPE}"
)

# Kaggle working directory.
WORKING_DIR = Path("/kaggle/working")

# Model-specific output directory.
MODEL_OUTPUT_DIR = WORKING_DIR / "outputs" / MODEL_NAME
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Normal output filenames.
SUMMARY_CSV_PATH = MODEL_OUTPUT_DIR / f"{MODEL_NAME}_summary.csv"
PREDICTIONS_CSV_PATH = MODEL_OUTPUT_DIR / "predictions.csv"
HISTORY_CSV_PATH = MODEL_OUTPUT_DIR / "training_history.csv"
CONFIG_JSON_PATH = MODEL_OUTPUT_DIR / "config.json"
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / "best_model.pth"
ZIP_PATH = MODEL_OUTPUT_DIR / "kaggle_working.zip"

print("Output paths created.")
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Model output directory: {MODEL_OUTPUT_DIR}")
print(f"Summary CSV path: {SUMMARY_CSV_PATH}")


# # Dataset

# ## dataset paths

# In[ ]:


# ============================================================
# 6. KAGGLE DATASET PATH SETUP
# ============================================================
# Add all possible Kaggle input dataset paths here.
# If a path does not exist in a Kaggle notebook, the scanner will simply skip it.

KAGGLE_INPUT_DIR = Path("/kaggle/input")

DATASET_PATHS = {
    # DMImageDetect datasets
    "dmimagedetect_test": KAGGLE_INPUT_DIR / "dmimagedetect-test",
    "dmimagedetect_train": KAGGLE_INPUT_DIR / "dmimagedetect-traintest",
    "dmimagedetect_realraise": KAGGLE_INPUT_DIR / "dmimagedetect-realraise",

    # Corvi latent diffusion dataset
    "corvi_latent_diffusion": KAGGLE_INPUT_DIR / "corvi-latent-diffusion-trainingset",

    # Wang CNNDetection dataset
    "wang_cnndetection": KAGGLE_INPUT_DIR / "wang-cnndetection-dataset",

    # Optional weights dataset
    "clipdet_weights": KAGGLE_INPUT_DIR / "dmimagedetect-clipdetweights",
}

print("Checking available Kaggle datasets...\n")

for key, path in DATASET_PATHS.items():
    exists = path.exists()
    print(f"{key:30s} -> {path} | exists = {exists}")


# ## helper functions to load, build

# In[ ]:


# ============================================================
# 7. DATASET SCANNING HELPERS
# ============================================================
# These functions create a dataframe of image paths and labels.
# The dataframe format will be shared across all model notebooks.

REAL_KEYWORDS = [
    "real",
    "authentic",
    "human",
    "nature",
    "raise"
]

SYNTHETIC_KEYWORDS = [
    "fake",
    "synthetic",
    "ai",
    "generated",
    "gan",
    "progan",
    "stylegan",
    "stylegan2",
    "biggan",
    "cyclegan",
    "stargan",
    "gaugan",
    "diffusion",
    "latent",
    "ldm",
    "glide",
    "dalle",
    "stable",
    "midjourney"
]


def is_image_file(path):
    """
    Check whether a file path has an image extension.

    Parameters:
        path (Path): File path.

    Returns:
        bool: True if the file looks like an image file.
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_label_from_path(path):
    """
    Infer image label from folder/file path.

    Label convention:
        0 = real
        1 = synthetic

    This function uses folder names and file paths.
    If the label cannot be guessed, it returns None.

    Parameters:
        path (Path): Image path.

    Returns:
        int or None: 0 for real, 1 for synthetic, None if unknown.
    """
    path_text = str(path).lower()

    # Check synthetic keywords first because some paths may contain words like "realistic".
    for keyword in SYNTHETIC_KEYWORDS:
        if keyword in path_text:
            return 1

    for keyword in REAL_KEYWORDS:
        if keyword in path_text:
            return 0

    return None


def infer_generator_from_path(path):
    """
    Try to infer the generator/source type from the image path.
    This is useful for later error analysis.

    Parameters:
        path (Path): Image path.

    Returns:
        str: Generator/source name if guessed, otherwise "unknown".
    """
    path_text = str(path).lower()

    generator_keywords = [
        "progan", "stylegan", "stylegan2", "biggan", "cyclegan", "stargan",
        "gaugan", "diffusion", "latent", "ldm", "glide", "dalle",
        "stable", "midjourney", "real", "raise"
    ]

    for keyword in generator_keywords:
        if keyword in path_text:
            return keyword

    return "unknown"


def scan_dataset(dataset_key, dataset_path, max_images=None):
    """
    Scan one dataset folder and return a dataframe with image paths and labels.

    Parameters:
        dataset_key (str): Short name of the dataset.
        dataset_path (Path): Path to the dataset folder.
        max_images (int or None): Optional image limit for quick debugging.

    Returns:
        pd.DataFrame: Dataframe with filepath, label, dataset, generator, and image name.
    """
    rows = []

    if not dataset_path.exists():
        print(f"[SKIP] Dataset path does not exist: {dataset_path}")
        return pd.DataFrame(columns=["filepath", "image_name", "label", "source_dataset", "generator"])

    image_paths = [p for p in dataset_path.rglob("*") if p.is_file() and is_image_file(p)]

    if max_images is not None:
        image_paths = image_paths[:max_images]

    for img_path in image_paths:
        label = infer_label_from_path(img_path)

        # If label cannot be inferred, we skip for training/evaluation.
        # For test inference later, we can allow label=None separately.
        if label is None:
            continue

        rows.append({
            "filepath": str(img_path),
            "image_name": img_path.name,
            "label": label,
            "source_dataset": dataset_key,
            "generator": infer_generator_from_path(img_path)
        })

    df = pd.DataFrame(rows)

    print(f"[SCAN] {dataset_key}")
    print(f"  Path: {dataset_path}")
    print(f"  Images found with labels: {len(df)}")

    if len(df) > 0:
        print(df["label"].value_counts().rename(index={0: "real", 1: "synthetic"}))

    return df


def print_dataset_summary(df, name):
    """
    Print a clean summary for a dataset dataframe.

    Parameters:
        df (pd.DataFrame): Dataset dataframe.
        name (str): Display name.
    """
    print("\n" + "=" * 60)
    print(f"DATASET SUMMARY: {name}")
    print("=" * 60)

    if df is None or len(df) == 0:
        print("No images found.")
        return

    print(f"Total images: {len(df)}")
    print("\nLabel counts:")
    print(df["label"].value_counts().rename(index={0: "real", 1: "synthetic"}))

    print("\nSource dataset counts:")
    print(df["source_dataset"].value_counts())

    print("\nGenerator/source counts:")
    print(df["generator"].value_counts().head(20))


# ## load dataset

# In[ ]:


# ============================================================
# 8. BUILD TRAINING AND EVALUATION DATAFRAMES
# ============================================================
# Training data is selectable.
# Evaluation data should stay fixed for all models.

train_dfs = []

# Add Wang dataset to training if selected.
if USE_WANG_TRAIN:
    train_dfs.append(
        scan_dataset(
            dataset_key="wang_cnndetection",
            dataset_path=DATASET_PATHS["wang_cnndetection"],
            max_images=MAX_TRAIN_IMAGES
        )
    )

# Add Corvi dataset to training if selected.
if USE_CORVI_TRAIN:
    train_dfs.append(
        scan_dataset(
            dataset_key="corvi_latent_diffusion",
            dataset_path=DATASET_PATHS["corvi_latent_diffusion"],
            max_images=MAX_TRAIN_IMAGES
        )
    )

# Add DMImageDetect train dataset if selected.
if USE_DMIMAGEDETECT_TRAIN:
    train_dfs.append(
        scan_dataset(
            dataset_key="dmimagedetect_train",
            dataset_path=DATASET_PATHS["dmimagedetect_train"],
            max_images=MAX_TRAIN_IMAGES
        )
    )

# Add RealRAISE if selected.
if USE_REALRAISE_TRAIN:
    train_dfs.append(
        scan_dataset(
            dataset_key="dmimagedetect_realraise",
            dataset_path=DATASET_PATHS["dmimagedetect_realraise"],
            max_images=MAX_TRAIN_IMAGES
        )
    )

# Combine all selected training datasets.
if len(train_dfs) > 0:
    train_df = pd.concat(train_dfs, ignore_index=True)
else:
    train_df = pd.DataFrame(columns=["filepath", "image_name", "label", "source_dataset", "generator"])

# Remove duplicate filepaths just in case the same data appears in multiple folders.
train_df = train_df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

# Build fixed evaluation dataframe.
eval_df = scan_dataset(
    dataset_key=EVALUATION_DATASET_KEY,
    dataset_path=DATASET_PATHS[EVALUATION_DATASET_KEY],
    max_images=MAX_EVAL_IMAGES
)

eval_df = eval_df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

print_dataset_summary(train_df, "TRAINING DATA")
print_dataset_summary(eval_df, "FIXED EVALUATION DATA")


# In[ ]:


# ============================================================
# 17. TRAIN/EVALUATION LEAKAGE SAFETY CHECK
# ============================================================
# This makes sure the same exact image path is not used for both training and evaluation.

if REMOVE_EVAL_FROM_TRAIN:
    before_count = len(train_df)

    eval_paths = set(eval_df["filepath"].tolist())
    train_df = train_df[~train_df["filepath"].isin(eval_paths)].reset_index(drop=True)

    after_count = len(train_df)
    removed_count = before_count - after_count

    print("Train/evaluation leakage check completed.")
    print(f"Training images before removal: {before_count}")
    print(f"Training images after removal:  {after_count}")
    print(f"Removed overlapping images:      {removed_count}")
else:
    print("Leakage removal is disabled. Make sure train/eval datasets are separate.")


# In[ ]:


# ============================================================
# 9. OPTIONAL DATASET CHECK
# ============================================================
# This cell helps us quickly see whether labels were inferred correctly.

print("Training dataframe preview:")
display(train_df.head())

print("\nEvaluation dataframe preview:")
display(eval_df.head())

# Check if we have both classes in training and evaluation.
if len(train_df) > 0:
    train_labels = set(train_df["label"].unique())
    print(f"\nTrain labels found: {train_labels}")
    if train_labels != {0, 1}:
        print("WARNING: Training set does not contain both real and synthetic labels.")

if len(eval_df) > 0:
    eval_labels = set(eval_df["label"].unique())
    print(f"Evaluation labels found: {eval_labels}")
    if eval_labels != {0, 1}:
        print("WARNING: Evaluation set does not contain both real and synthetic labels.")


# # functions

# ## metric calculation

# In[ ]:


# ============================================================
# 10. METRIC CALCULATION FUNCTIONS
# ============================================================
# These functions are shared across all models.
# They calculate the competition-relevant metrics for Task A.

def calculate_eer(y_true, y_prob):
    """
    Calculate Equal Error Rate (EER).

    EER is the point where false positive rate and false negative rate are approximately equal.

    Parameters:
        y_true (array-like): True labels, 0 for real and 1 for synthetic.
        y_prob (array-like): Predicted probability for synthetic class.

    Returns:
        float: Equal Error Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # False negative rate is 1 - true positive rate.
    fnr = 1 - tpr

    # Find the point where FPR and FNR are closest.
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2

    return float(eer)


def calculate_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate binary classification metrics.

    Parameters:
        y_true (array-like): True labels, 0 for real and 1 for synthetic.
        y_prob (array-like): Predicted probability for synthetic class.
        threshold (float): Decision threshold.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, AUC, AP, EER, and confusion matrix values.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)

    # Convert probabilities into predicted labels using threshold.
    y_pred = (y_prob >= threshold).astype(int)

    # Basic metrics.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Some metrics require both classes to exist.
    # If only one class exists, sklearn can throw an error, so we handle that safely.
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    try:
        avg_precision = average_precision_score(y_true, y_prob)
    except ValueError:
        avg_precision = np.nan

    try:
        eer = calculate_eer(y_true, y_prob)
    except ValueError:
        eer = np.nan

    # Confusion matrix values.
    # Labels are fixed as [0, 1] so output order is always:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "average_precision": float(avg_precision) if not np.isnan(avg_precision) else np.nan,
        "eer": float(eer) if not np.isnan(eer) else np.nan,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    return metrics


def find_best_f1_threshold(y_true, y_prob, thresholds=None):
    """
    Find the threshold that gives the best F1 score.

    Parameters:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities.
        thresholds (list or None): Thresholds to test.

    Returns:
        tuple: best_threshold, best_f1
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


print("Metric functions ready.")


# ## logging

# In[ ]:


# ============================================================
# 11. SAVING AND LOGGING FUNCTIONS
# ============================================================
# These functions save predictions, training history, config, and summary rows.

def save_config_json(config_path):
    """
    Save the current experiment settings into config.json.

    Parameters:
        config_path (Path): Path where config JSON should be saved.
    """
    config = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": TIMESTAMP,
        "runner": RUNNER_NAME,
        "model_name": MODEL_NAME,
        "run_type": RUN_TYPE,

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": SCHEDULER_NAME,
        "loss_function": LOSS_FUNCTION_NAME,

        "image_size": IMAGE_SIZE,
        "pretrained": PRETRAINED,
        "freeze_backbone": FREEZE_BACKBONE,
        "augmentation_type": AUGMENTATION_TYPE,
        "seed": SEED,
        "threshold": THRESHOLD,

        "use_wang_train": USE_WANG_TRAIN,
        "use_corvi_train": USE_CORVI_TRAIN,
        "use_dmimagedetect_train": USE_DMIMAGEDETECT_TRAIN,
        "use_realraise_train": USE_REALRAISE_TRAIN,
        "evaluation_dataset_key": EVALUATION_DATASET_KEY,

        "max_train_images": MAX_TRAIN_IMAGES,
        "max_eval_images": MAX_EVAL_IMAGES,

        "experiment_notes": EXPERIMENT_NOTES
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to: {config_path}")


def save_predictions_csv(eval_dataframe, y_prob, threshold, predictions_path):
    """
    Save latest evaluation predictions to predictions.csv.

    Parameters:
        eval_dataframe (pd.DataFrame): Evaluation dataframe with filepath and true label.
        y_prob (array-like): Predicted probability for synthetic class.
        threshold (float): Decision threshold.
        predictions_path (Path): Output CSV path.

    Returns:
        pd.DataFrame: Predictions dataframe.
    """
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    pred_df = eval_dataframe.copy()
    pred_df["experiment_id"] = EXPERIMENT_ID
    pred_df["runner"] = RUNNER_NAME
    pred_df["model_name"] = MODEL_NAME
    pred_df["true_label"] = pred_df["label"].astype(int)
    pred_df["pred_prob"] = y_prob
    pred_df["threshold"] = threshold
    pred_df["pred_label"] = y_pred
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    # Reorder important columns first.
    first_cols = [
        "experiment_id",
        "runner",
        "model_name",
        "filepath",
        "image_name",
        "true_label",
        "pred_prob",
        "threshold",
        "pred_label",
        "correct",
        "source_dataset",
        "generator"
    ]

    remaining_cols = [col for col in pred_df.columns if col not in first_cols]
    pred_df = pred_df[first_cols + remaining_cols]

    pred_df.to_csv(predictions_path, index=False)

    print(f"Predictions saved to: {predictions_path}")
    print(f"Total predictions saved: {len(pred_df)}")

    return pred_df


def save_training_history_csv(history, history_path):
    """
    Save latest training history to training_history.csv.

    Parameters:
        history (list of dict): One dictionary per epoch.
        history_path (Path): Output CSV path.

    Returns:
        pd.DataFrame: History dataframe.
    """
    history_df = pd.DataFrame(history)

    if len(history_df) > 0:
        history_df["experiment_id"] = EXPERIMENT_ID
        history_df["runner"] = RUNNER_NAME
        history_df["model_name"] = MODEL_NAME

        # Put important columns first.
        first_cols = ["experiment_id", "runner", "model_name"]
        remaining_cols = [col for col in history_df.columns if col not in first_cols]
        history_df = history_df[first_cols + remaining_cols]

    history_df.to_csv(history_path, index=False)

    print(f"Training history saved to: {history_path}")
    print(f"History rows saved: {len(history_df)}")

    return history_df


def append_summary_row(summary_path, metrics, best_epoch=None, best_val_loss=None):
    """
    Append one row to the model summary CSV.

    Parameters:
        summary_path (Path): Model summary CSV path.
        metrics (dict): Final evaluation metrics.
        best_epoch (int or None): Best epoch number.
        best_val_loss (float or None): Best validation loss.

    Returns:
        pd.DataFrame: Full updated summary dataframe.
    """
    train_real_count = int((train_df["label"] == 0).sum()) if len(train_df) > 0 else 0
    train_synth_count = int((train_df["label"] == 1).sum()) if len(train_df) > 0 else 0
    eval_real_count = int((eval_df["label"] == 0).sum()) if len(eval_df) > 0 else 0
    eval_synth_count = int((eval_df["label"] == 1).sum()) if len(eval_df) > 0 else 0

    train_datasets_used = ",".join(sorted(train_df["source_dataset"].unique())) if len(train_df) > 0 else ""

    row = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": TIMESTAMP,
        "runner": RUNNER_NAME,
        "model_name": MODEL_NAME,
        "run_type": RUN_TYPE,

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": SCHEDULER_NAME,
        "loss_function": LOSS_FUNCTION_NAME,

        "image_size": IMAGE_SIZE,
        "pretrained": PRETRAINED,
        "freeze_backbone": FREEZE_BACKBONE,
        "augmentation_type": AUGMENTATION_TYPE,
        "seed": SEED,

        "train_datasets_used": train_datasets_used,
        "evaluation_dataset": EVALUATION_DATASET_KEY,
        "num_train_images": len(train_df),
        "num_eval_images": len(eval_df),
        "real_train_count": train_real_count,
        "synthetic_train_count": train_synth_count,
        "real_eval_count": eval_real_count,
        "synthetic_eval_count": eval_synth_count,

        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,

        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "eer": metrics.get("eer"),
        "threshold": metrics.get("threshold"),

        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),

        "notes": EXPERIMENT_NOTES
    }

    row_df = pd.DataFrame([row])

    # If summary CSV already exists, append to it.
    if summary_path.exists():
        old_summary_df = pd.read_csv(summary_path)
        updated_summary_df = pd.concat([old_summary_df, row_df], ignore_index=True)
    else:
        updated_summary_df = row_df

    updated_summary_df.to_csv(summary_path, index=False)

    print(f"Summary row appended to: {summary_path}")
    print(f"Total experiments logged for {MODEL_NAME}: {len(updated_summary_df)}")

    return updated_summary_df


print("Saving and logging functions ready.")


# # data preprocessing

# ## image transform

# In[ ]:


# ============================================================
# 18. IMAGE TRANSFORMS
# ============================================================
# These transforms prepare images for ConvNeXt-Tiny.
# ConvNeXt pretrained weights expect ImageNet-style normalization.

# ImageNet mean and std are used because ConvNeXt-Tiny was pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(augmentation_type):
    """
    Create training transforms based on augmentation type.

    Parameters:
        augmentation_type (str): Type of augmentation to use.

    Returns:
        torchvision.transforms.Compose: Training transform pipeline.
    """

    if augmentation_type == "basic":
        # Simple baseline transform.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif augmentation_type == "light_aug":
        # Slightly stronger augmentation.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif augmentation_type == "jpeg_like":
        # Approximation of real-world resizing/cropping effects.
        # JPEG compression itself is not added here to keep it simple and fast.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 48, IMAGE_SIZE + 48)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.20),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    else:
        raise ValueError(f"Unknown AUGMENTATION_TYPE: {augmentation_type}")


def get_eval_transforms():
    """
    Create evaluation transforms.

    Returns:
        torchvision.transforms.Compose: Evaluation transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


train_transforms = get_train_transforms(AUGMENTATION_TYPE)
eval_transforms = get_eval_transforms()

print("Transforms created.")
print(f"Training augmentation type: {AUGMENTATION_TYPE}")


# ## pytorch

# In[ ]:


# ============================================================
# 19. PYTORCH IMAGE DATASET CLASS
# ============================================================
# This class reads image paths from train_df/eval_df and returns tensors + labels.

class BinaryImageDataset(Dataset):
    """
    Dataset for binary real vs synthetic image classification.

    Each item returns:
        image tensor
        label tensor
        image path
    """

    def __init__(self, dataframe, transform=None):
        """
        Parameters:
            dataframe (pd.DataFrame): Dataframe with filepath and label columns.
            transform: Torchvision transforms.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Load one image and its label.

        Parameters:
            idx (int): Row index.

        Returns:
            image (Tensor): Transformed image.
            label (Tensor): Float label, 0.0 for real and 1.0 for synthetic.
            filepath (str): Original image path.
        """
        row = self.dataframe.iloc[idx]
        image_path = row["filepath"]
        label = float(row["label"])

        # Convert image to RGB so all inputs have 3 channels.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # BCEWithLogitsLoss expects float labels.
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, image_path


print("BinaryImageDataset class ready.")


# ## dataloaders

# In[ ]:


# ============================================================
# 20. CREATE DATASETS AND DATALOADERS
# ============================================================
# DataLoaders feed batches of images into the model.

train_dataset = BinaryImageDataset(train_df, transform=train_transforms)
eval_dataset = BinaryImageDataset(eval_df, transform=eval_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print("Datasets and DataLoaders created.")
print(f"Training images:   {len(train_dataset)}")
print(f"Evaluation images: {len(eval_dataset)}")
print(f"Training batches:  {len(train_loader)}")
print(f"Evaluation batches:{len(eval_loader)}")


# # model

# ## create model

# In[ ]:


# ============================================================
# 21. CREATE CONVNEXT-TINY MODEL
# ============================================================
# We use pretrained ConvNeXt-Tiny and replace the final classifier for binary classification.

def create_convnext_tiny(pretrained=True, freeze_backbone=False):
    """
    Create ConvNeXt-Tiny for binary real vs synthetic image classification.

    Parameters:
        pretrained (bool): Whether to load ImageNet pretrained weights.
        freeze_backbone (bool): Whether to freeze feature extractor layers.

    Returns:
        torch.nn.Module: ConvNeXt-Tiny model.
    """

    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        print("Loaded ConvNeXt-Tiny with ImageNet pretrained weights.")
    else:
        model = models.convnext_tiny(weights=None)
        print("Loaded ConvNeXt-Tiny without pretrained weights.")

    # Freeze backbone if selected.
    # ConvNeXt has a feature extractor stored in model.features.
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only classifier will train.")
    else:
        print("Backbone is trainable.")

    # ConvNeXt-Tiny classifier usually looks like:
    # Sequential(
    #   (0): LayerNorm2d(...)
    #   (1): Flatten(...)
    #   (2): Linear(in_features=768, out_features=1000)
    # )
    in_features = model.classifier[2].in_features

    # Replace final layer with one output logit.
    # One logit is enough for binary classification with BCEWithLogitsLoss.
    model.classifier[2] = nn.Linear(in_features, 1)

    return model


model = create_convnext_tiny(
    pretrained=PRETRAINED,
    freeze_backbone=FREEZE_BACKBONE
)

model = model.to(DEVICE)

print("Model created and moved to device.")
print(model.classifier)


# ## optimizers, schedulers

# In[ ]:


# ============================================================
# 22. LOSS FUNCTION, OPTIMIZER, AND SCHEDULER
# ============================================================
# This cell creates the training components.

def create_optimizer(model, optimizer_name):
    """
    Create optimizer based on selected optimizer name.

    Parameters:
        model (torch.nn.Module): Model to train.
        optimizer_name (str): Optimizer name.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """

    # Only train parameters where requires_grad=True.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, scheduler_name):
    """
    Create learning rate scheduler.

    Parameters:
        optimizer: PyTorch optimizer.
        scheduler_name (str): Scheduler name.

    Returns:
        scheduler or None
    """

    if scheduler_name.lower() == "none":
        return None

    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.5
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


criterion = nn.BCEWithLogitsLoss()
optimizer = create_optimizer(model, OPTIMIZER_NAME)
scheduler = create_scheduler(optimizer, SCHEDULER_NAME)

print("Loss, optimizer, and scheduler created.")
print(f"Loss function: {LOSS_FUNCTION_NAME}")
print(f"Optimizer: {OPTIMIZER_NAME}")
print(f"Scheduler: {SCHEDULER_NAME}")


# # training

# ## functions

# In[ ]:


# ============================================================
# 23. TRAINING AND EVALUATION FUNCTIONS
# ============================================================
# These functions handle one training epoch and one evaluation pass.

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Train the model for one epoch.

    Parameters:
        model: PyTorch model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CUDA or CPU.
        scaler: GradScaler for mixed precision.

    Returns:
        float: Average training loss.
    """

    model.train()
    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels, paths in progress_bar:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        # Mixed precision speeds up training on GPU.
        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            # Optional gradient clipping.
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Optional gradient clipping.
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = running_loss / max(total_samples, 1)
    return avg_loss


def evaluate_model(model, loader, criterion, device):
    """
    Evaluate model on validation/evaluation dataset.

    Parameters:
        model: PyTorch model.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: CUDA or CPU.

    Returns:
        tuple: avg_loss, y_true, y_prob, image_paths
    """

    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_paths = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels, paths in progress_bar:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)

            # Convert logits to probabilities for synthetic class.
            probs = torch.sigmoid(logits).view(-1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.view(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(list(paths))

            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = running_loss / max(total_samples, 1)

    return avg_loss, np.array(all_labels), np.array(all_probs), all_paths


print("Training and evaluation functions ready.")


# ## run training

# In[ ]:


# ============================================================
# 24. MAIN TRAINING LOOP
# ============================================================
# This trains EfficientNet-B0 and saves the best model based on validation F1.

history = []

best_epoch = None
best_val_loss = float("inf")
best_val_f1 = -1.0

# GradScaler is used only when AMP is enabled and CUDA is available.
scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.type == "cuda") else None

print("=" * 70)
print("STARTING CONVNEXT-TINY TRAINING")
print("=" * 70)
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Training images: {len(train_dataset)}")
print(f"Evaluation images: {len(eval_dataset)}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Optimizer: {OPTIMIZER_NAME}")
print(f"Augmentation: {AUGMENTATION_TYPE}")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    # Train for one epoch.
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        scaler=scaler
    )

    # Evaluate after each epoch.
    val_loss, epoch_y_true, epoch_y_prob, _ = evaluate_model(
        model=model,
        loader=eval_loader,
        criterion=criterion,
        device=DEVICE
    )

    # Find best F1 threshold for this epoch.
    epoch_best_threshold, epoch_best_threshold_f1 = find_best_f1_threshold(
        y_true=epoch_y_true,
        y_prob=epoch_y_prob
    )

    # Calculate metrics using the best threshold for this epoch.
    epoch_metrics = calculate_binary_metrics(
        y_true=epoch_y_true,
        y_prob=epoch_y_prob,
        threshold=epoch_best_threshold
    )

    # Step scheduler after validation.
    if scheduler is not None:
        scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    # Save epoch history.
    epoch_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": epoch_metrics["accuracy"],
        "val_precision": epoch_metrics["precision"],
        "val_recall": epoch_metrics["recall"],
        "val_f1": epoch_metrics["f1"],
        "val_roc_auc": epoch_metrics["roc_auc"],
        "val_average_precision": epoch_metrics["average_precision"],
        "val_eer": epoch_metrics["eer"],
        "best_threshold_for_epoch": epoch_best_threshold,
        "learning_rate": current_lr
    }

    history.append(epoch_row)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val F1: {epoch_metrics['f1']:.4f} | "
        f"Val Acc: {epoch_metrics['accuracy']:.4f} | "
        f"Val AUC: {epoch_metrics['roc_auc']:.4f} | "
        f"Best Threshold: {epoch_best_threshold:.2f}"
    )

    # Decide if this epoch is the best one.
    if SAVE_BEST_BY == "f1":
        is_best = epoch_metrics["f1"] > best_val_f1
    elif SAVE_BEST_BY == "val_loss":
        is_best = val_loss < best_val_loss
    else:
        raise ValueError(f"Unknown SAVE_BEST_BY: {SAVE_BEST_BY}")

    # Save best model.
    if is_best:
        best_epoch = epoch
        best_val_loss = val_loss
        best_val_f1 = epoch_metrics["f1"]

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(f"New best model saved at epoch {epoch}.")
        print(f"Best model path: {BEST_MODEL_PATH}")

print("\nTraining completed.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation F1: {best_val_f1:.4f}")


# # evaluation

# In[ ]:


# ============================================================
# 25. LOAD BEST MODEL AND FINAL EVALUATION
# ============================================================
# We reload the best saved model and evaluate it once more for final logging.

if BEST_MODEL_PATH.exists():
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Loaded best model from: {BEST_MODEL_PATH}")
else:
    print("WARNING: Best model file not found. Using current model weights.")

final_val_loss, y_true, y_prob, image_paths = evaluate_model(
    model=model,
    loader=eval_loader,
    criterion=criterion,
    device=DEVICE
)

# Find best threshold based on final evaluation F1.
best_threshold, best_threshold_f1 = find_best_f1_threshold(
    y_true=y_true,
    y_prob=y_prob
)

# Calculate final metrics with best threshold.
metrics = calculate_binary_metrics(
    y_true=y_true,
    y_prob=y_prob,
    threshold=best_threshold
)

print("=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)
print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Best threshold:        {best_threshold:.2f}")
print(f"Accuracy:              {metrics['accuracy']:.4f}")
print(f"Precision:             {metrics['precision']:.4f}")
print(f"Recall:                {metrics['recall']:.4f}")
print(f"F1 Score:              {metrics['f1']:.4f}")
print(f"ROC AUC:               {metrics['roc_auc']:.4f}")
print(f"Average Precision:     {metrics['average_precision']:.4f}")
print(f"EER:                   {metrics['eer']:.4f}")
print(f"TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TP: {metrics['tp']}")
print("=" * 70)


# # log, save, analyze

# In[ ]:


# ============================================================
# 26. SAVE OUTPUTS AND APPEND SUMMARY
# ============================================================
# This saves:
# - config.json
# - predictions.csv
# - training_history.csv
# - efficientnet_b0_summary.csv

# Save latest config.
save_config_json(CONFIG_JSON_PATH)

# Save latest predictions.
predictions_df = save_predictions_csv(
    eval_dataframe=eval_df,
    y_prob=y_prob,
    threshold=best_threshold,
    predictions_path=PREDICTIONS_CSV_PATH
)

# Save latest training history.
history_df = save_training_history_csv(
    history=history,
    history_path=HISTORY_CSV_PATH
)

# Append one row to model summary CSV.
summary_df = append_summary_row(
    summary_path=SUMMARY_CSV_PATH,
    metrics=metrics,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss
)

print("\nLatest summary rows:")
display(summary_df.tail())


# In[ ]:


# ============================================================
# 27. QUICK ERROR ANALYSIS
# ============================================================
# This helps us quickly see what the model got wrong.

wrong_df = predictions_df[predictions_df["correct"] == False].copy()

print(f"Total wrong predictions: {len(wrong_df)} out of {len(predictions_df)}")

if len(wrong_df) > 0:
    print("\nWrong prediction counts by true label:")
    print(wrong_df["true_label"].value_counts().rename(index={0: "real", 1: "synthetic"}))

    print("\nWrong prediction counts by generator/source:")
    print(wrong_df["generator"].value_counts().head(20))

    print("\nSample wrong predictions:")
    display(wrong_df[[
        "image_name",
        "true_label",
        "pred_prob",
        "pred_label",
        "source_dataset",
        "generator",
        "filepath"
    ]].head(20))
else:
    print("No wrong predictions found.")


# # save outputs in a zip
# because the outputs are alot, instead of downloading each manually we can just download one zip and it would download in the correct folder structure and we can just paste in github/vscode

# In[ ]:


import os
import shutil
from IPython.display import FileLink, display as ipy_display

# Path you want to zip
source_folder = "/kaggle/working"

# Where the zip file will be created
zip_base_path = "/kaggle/working/kaggle_working_backup"

# Create kaggle_working_backup.zip
# shutil.make_archive expects the path without ".zip"
zip_file = shutil.make_archive(zip_base_path, "zip", source_folder)

print("ZIP created at:", zip_file)

# Show a clickable download link inside the notebook
ipy_display(FileLink(zip_file))

