#!/usr/bin/env python
# coding: utf-8

# # 00 - Shared Experiment Template
# 
# This notebook contains the common code used by all model experiment notebooks.
# 
# It includes:
# - shared parameter block
# - dataset path setup
# - dataset scanning
# - fixed evaluation dataset setup
# - metric calculation functions
# - summary CSV logging
# - predictions saving
# - training history saving
# - config JSON saving
# - Kaggle working zip creation
# 
# Model-specific code should be added after this template is copied into a model notebook.

# In[15]:


# ============================================================
# 13. MODEL-SPECIFIC CODE GOES BELOW THIS POINT
# ============================================================
# When making a real model notebook:
#
# 1. Duplicate this notebook.
# 2. Rename it, for example:
#       01_efficientnet_b0_experiments.ipynb
#
# 3. Change the top parameter block:
#       MODEL_NAME = "efficientnet_b0"
#
# 4. Add model-specific code below "# functions":
#       - torchvision/timm imports
#       - transforms
#       - Dataset class
#       - DataLoaders
#       - model creation
#       - loss function
#       - optimizer
#       - training loop
#       - evaluation loop
#
# 5. At the end of the model notebook, call:
#       save_config_json(CONFIG_JSON_PATH)
#       save_predictions_csv(eval_df, y_prob, best_threshold, PREDICTIONS_CSV_PATH)
#       save_training_history_csv(history, HISTORY_CSV_PATH)
#       append_summary_row(SUMMARY_CSV_PATH, metrics, best_epoch, best_val_loss)


# In[16]:


# ============================================================
# 14. EXAMPLE FINAL SAVING CALLS
# ============================================================
# This cell is only an example.
# Do not run this in the sample template unless y_prob/history/metrics exist.

"""
# Example after model evaluation:

# Find best threshold based on evaluation F1.
best_threshold, best_threshold_f1 = find_best_f1_threshold(
    y_true=eval_df["label"].values,
    y_prob=y_prob
)

# Calculate final metrics using best threshold.
metrics = calculate_binary_metrics(
    y_true=eval_df["label"].values,
    y_prob=y_prob,
    threshold=best_threshold
)

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

# Show latest summary.
display(summary_df.tail())
"""


# # Imports

# In[17]:


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


# In[18]:


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

# In[19]:


# ============================================================
# 1. SHARED PARAMETER BLOCK
# ============================================================
# Change these values before each experiment run.
# All important experiment settings should stay here so nobody has to search inside the code.

RUNNER_NAME = "zuha"                 # Change to: "zuha", "izma", or "fatima"
MODEL_NAME = "sample_model"          # Example: "efficientnet_b0", "convnext_tiny", "clip_vit"
RUN_TYPE = "constrained"             # Use "constrained" or "open"

# General training settings
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OPTIMIZER_NAME = "adamw"
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "none"
LOSS_FUNCTION_NAME = "BCEWithLogitsLoss"

# Image/model settings
IMAGE_SIZE = 224
PRETRAINED = True
FREEZE_BACKBONE = False
AUGMENTATION_TYPE = "basic"

# Binary classification settings
# Label convention:
# 0 = real
# 1 = synthetic
THRESHOLD = 0.5

# Notes for this experiment.
# Write what changed in this run, for example:
# "baseline run", "increased epochs", "lower lr", "jpeg augmentation", etc.
EXPERIMENT_NOTES = "sample template run"

print("Shared parameters loaded.")
print(f"Runner: {RUNNER_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"Run type: {RUN_TYPE}")


# In[20]:


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
USE_DMIMAGEDETECT_TRAIN = True
USE_REALRAISE_TRAIN = False

# ----------------------------
# Fixed evaluation dataset
# ----------------------------
# IMPORTANT:
# Keep this same across all model notebooks unless the whole team agrees to change it.
# This is how we compare EfficientNet vs ConvNeXt vs CLIP fairly.
EVALUATION_DATASET_KEY = "dmimagedetect_train"   # Change later if official validation folder is separate

# Limit images for quick debugging.
# Set to None for full experiment.
MAX_TRAIN_IMAGES = None
MAX_EVAL_IMAGES = None

# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

print("Dataset selection loaded.")
print("Training dataset choices:")
print(f"  Wang train: {USE_WANG_TRAIN}")
print(f"  Corvi train: {USE_CORVI_TRAIN}")
print(f"  DMImageDetect train: {USE_DMIMAGEDETECT_TRAIN}")
print(f"  RealRAISE train: {USE_REALRAISE_TRAIN}")
print(f"Fixed evaluation dataset key: {EVALUATION_DATASET_KEY}")


# In[21]:


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

# In[22]:


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

# In[23]:


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

# In[24]:


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


# In[25]:


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

# In[26]:


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

# In[27]:


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


# # ADD MODEL CODE HERE AND DELETE THIS MARKDOWN

# # save outputs in a zip
# because the outputs are alot, instead of downloading each manually we can just download one zip and it would download in the correct folder structure and we can just paste in github/vscode

# In[28]:


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

