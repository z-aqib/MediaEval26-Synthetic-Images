#!/usr/bin/env python
# coding: utf-8

# # Experiment Analysis Notebook
# 
# This notebook analyzes all Task A model experiments.
# 
# It reads experiment summary CSVs from:
# - EfficientNet-B0
# - ConvNeXt-Tiny
# - CLIP/ViT
# - Optional baseline model
# 
# It creates:
# - one combined master summary CSV
# - best experiment ranking
# - model comparison plots
# - parameter effect plots
# - best model extraction from Git history where possible
# 
# This notebook is meant to run locally on CPU in VS Code.

# # imports

# In[1]:


# ============================================================
# 1. IMPORTS AND GLOBAL SETTINGS
# ============================================================
# This notebook runs locally on CPU.
# It does not train models and does not need GPU.

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Imports completed.")


# # variables

# In[2]:


# ============================================================
# 2. PROJECT PATH SETUP
# ============================================================
# Run this notebook from VS Code inside the repo.
# If the notebook is inside notebooks/, this will move one level up to repo root.

CURRENT_DIR = Path.cwd()

# If running from notebooks folder, repo root is parent.
if CURRENT_DIR.name == "notebooks":
    REPO_ROOT = CURRENT_DIR.parent
else:
    REPO_ROOT = CURRENT_DIR

OUTPUTS_DIR = REPO_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
BEST_OVERALL_DIR = OUTPUTS_DIR / "best_overall"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_OVERALL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Repo root: {REPO_ROOT}")
print(f"Outputs directory: {OUTPUTS_DIR}")
print(f"Analysis directory: {ANALYSIS_DIR}")
print(f"Plots directory: {PLOTS_DIR}")
print(f"Best overall directory: {BEST_OVERALL_DIR}")


# In[3]:


# ============================================================
# 3. MODEL FOLDERS TO ANALYZE
# ============================================================
# Add/remove model folders here if needed.

MODEL_FOLDERS = [
    "01_efficientnet_b0",
    "02_convnext_tiny",
    "03_clip_vit",
    "baseline_model",   # optional, included only if it has usable CSVs
]

# Main ranking metric.
# F1 is the most important metric for the competition.
PRIMARY_METRIC = "f1"

# Tie-breakers if F1 is same or missing.
TIE_BREAKER_METRICS = ["roc_auc", "accuracy"]

print("Model folders selected:")
for model_name in MODEL_FOLDERS:
    print(f"- {model_name}")


# # get experiments

# ## read summaries

# In[4]:


# ============================================================
# 4. READ MODEL SUMMARY CSV FILES
# ============================================================
# Each model folder should have one summary CSV:
# outputs/<model_name>/<model_name>_summary.csv

def find_summary_csv(model_folder):
    """
    Find the summary CSV for a model folder.

    Parameters:
        model_folder (str): Model folder name inside outputs/.

    Returns:
        Path or None: Summary CSV path if found.
    """
    model_dir = OUTPUTS_DIR / model_folder

    if not model_dir.exists():
        return None

    expected_path = model_dir / f"{model_folder}_summary.csv"

    if expected_path.exists():
        return expected_path

    # Fallback: find any file containing "summary" in name.
    candidates = list(model_dir.glob("*summary*.csv"))

    if len(candidates) > 0:
        return candidates[0]

    return None


def read_all_summary_csvs(model_folders):
    """
    Read all model summary CSVs and combine them into one dataframe.

    Parameters:
        model_folders (list): Model folder names.

    Returns:
        pd.DataFrame: Combined experiment summary dataframe.
    """
    all_dfs = []

    for model_folder in model_folders:
        summary_path = find_summary_csv(model_folder)

        if summary_path is None:
            print(f"[SKIP] No summary CSV found for: {model_folder}")
            continue

        print(f"[READ] {summary_path}")

        df = pd.read_csv(summary_path)

        # Add source path for traceability.
        df["summary_csv_path"] = str(summary_path)
        df["model_folder"] = model_folder

        # If model_name is missing, use folder name.
        if "model_name" not in df.columns:
            df["model_name"] = model_folder

        all_dfs.append(df)

    if len(all_dfs) == 0:
        print("No summary CSV files found.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    return combined_df


summary_df = read_all_summary_csvs(MODEL_FOLDERS)

print("\nCombined summary shape:", summary_df.shape)

if len(summary_df) > 0:
    display(summary_df.head())


# ## clean summaries

# In[5]:


# ============================================================
# 5. CLEAN AND STANDARDIZE SUMMARY DATA
# ============================================================
# This makes sure metric columns are numeric and missing columns do not break plots.

def clean_summary_dataframe(df):
    """
    Clean and standardize the combined summary dataframe.

    Parameters:
        df (pd.DataFrame): Combined summary dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()

    # Columns we expect to be numeric.
    numeric_columns = [
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "image_size",
        "num_train_images",
        "num_eval_images",
        "real_train_count",
        "synthetic_train_count",
        "real_eval_count",
        "synthetic_eval_count",
        "best_epoch",
        "best_val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "eer",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make sure key columns exist.
    required_text_columns = [
        "experiment_id",
        "timestamp",
        "runner",
        "model_name",
        "model_folder",
        "run_type",
        "optimizer",
        "scheduler",
        "augmentation_type",
        "train_datasets_used",
        "evaluation_dataset",
        "notes",
    ]

    for col in required_text_columns:
        if col not in df.columns:
            df[col] = ""

    # Sort by F1 if available.
    if "f1" in df.columns:
        df = df.sort_values(by="f1", ascending=False, na_position="last").reset_index(drop=True)

    return df


summary_df = clean_summary_dataframe(summary_df)

MASTER_SUMMARY_PATH = ANALYSIS_DIR / "all_experiments_summary.csv"
summary_df.to_csv(MASTER_SUMMARY_PATH, index=False)

print(f"Master summary saved to: {MASTER_SUMMARY_PATH}")
print(f"Total experiments found: {len(summary_df)}")

display(summary_df.head(10))


# # analyze

# ## rank best ACROSS ALL MODELS

# In[6]:


# ============================================================
# 6. BEST EXPERIMENTS OVERALL
# ============================================================
# We rank by F1 first, then ROC AUC, then accuracy.

def rank_experiments(df):
    """
    Rank experiments using F1, ROC AUC, and accuracy.

    Parameters:
        df (pd.DataFrame): Cleaned summary dataframe.

    Returns:
        pd.DataFrame: Ranked dataframe.
    """
    sort_columns = []
    ascending_values = []

    for metric in [PRIMARY_METRIC] + TIE_BREAKER_METRICS:
        if metric in df.columns:
            sort_columns.append(metric)
            ascending_values.append(False)

    if len(sort_columns) == 0:
        return df

    ranked_df = df.sort_values(
        by=sort_columns,
        ascending=ascending_values,
        na_position="last"
    ).reset_index(drop=True)

    ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)

    return ranked_df


ranked_df = rank_experiments(summary_df)

RANKED_SUMMARY_PATH = ANALYSIS_DIR / "ranked_experiments.csv"
ranked_df.to_csv(RANKED_SUMMARY_PATH, index=False)

print(f"Ranked experiments saved to: {RANKED_SUMMARY_PATH}")

important_cols = [
    "rank",
    "experiment_id",
    "model_name",
    "model_folder",
    "runner",
    "run_type",
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "scheduler",
    "augmentation_type",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "eer",
    "threshold",
    "notes",
]

existing_cols = [col for col in important_cols if col in ranked_df.columns]

display(ranked_df[existing_cols].head(20))


# ## rank best FOR EVERY MODEL

# In[7]:


# ============================================================
# 7. BEST EXPERIMENT PER MODEL
# ============================================================
# This shows the best row for each model according to F1.

if len(ranked_df) > 0:
    best_per_model_df = (
        ranked_df
        .sort_values(by=["model_folder", "f1", "roc_auc", "accuracy"], ascending=[True, False, False, False])
        .groupby("model_folder", as_index=False)
        .head(1)
        .sort_values(by="f1", ascending=False)
        .reset_index(drop=True)
    )

    BEST_PER_MODEL_PATH = ANALYSIS_DIR / "best_experiment_per_model.csv"
    best_per_model_df.to_csv(BEST_PER_MODEL_PATH, index=False)

    print(f"Best experiment per model saved to: {BEST_PER_MODEL_PATH}")
    display(best_per_model_df[existing_cols])
else:
    print("No experiments available.")


# # plot

# ## plot functions

# In[8]:


# ============================================================
# 8. PLOT HELPER FUNCTIONS
# ============================================================
# These helper functions save plots into outputs/analysis/plots.

def save_current_plot(filename):
    """
    Save the current matplotlib figure.

    Parameters:
        filename (str): Plot filename.
    """
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.show()


def plot_bar_metric_by_model(df, metric):
    """
    Plot best metric value per model.

    Parameters:
        df (pd.DataFrame): Summary dataframe.
        metric (str): Metric column to plot.
    """
    if metric not in df.columns:
        print(f"[SKIP] Metric not found: {metric}")
        return

    plot_df = (
        df.groupby("model_folder", as_index=False)[metric]
        .max()
        .sort_values(by=metric, ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model_folder"], plot_df[metric])
    plt.title(f"Best {metric.upper()} by Model")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=30, ha="right")
    save_current_plot(f"best_{metric}_by_model.png")


def plot_metric_over_experiments(df, metric):
    """
    Plot metric across experiment order for each model.

    Parameters:
        df (pd.DataFrame): Summary dataframe.
        metric (str): Metric column to plot.
    """
    if metric not in df.columns:
        print(f"[SKIP] Metric not found: {metric}")
        return

    plt.figure(figsize=(11, 6))

    for model_name, group_df in df.groupby("model_folder"):
        group_df = group_df.reset_index(drop=True)
        x_values = np.arange(1, len(group_df) + 1)
        plt.plot(x_values, group_df[metric], marker="o", label=model_name)

    plt.title(f"{metric.upper()} Across Experiments")
    plt.xlabel("Experiment Number Within Model")
    plt.ylabel(metric.upper())
    plt.legend()
    save_current_plot(f"{metric}_across_experiments.png")


print("Plot helper functions ready.")


# ## run plots ACROSS MODELS

# In[9]:


# ============================================================
# 9. MAIN MODEL COMPARISON PLOTS
# ============================================================
# These plots compare models using final metrics.

metrics_to_plot = [
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "average_precision",
    "eer",
]

for metric in metrics_to_plot:
    plot_bar_metric_by_model(ranked_df, metric)

for metric in ["f1", "accuracy", "roc_auc"]:
    plot_metric_over_experiments(summary_df, metric)


# ## run plots FOR EVERY MODEL

# In[10]:


# ============================================================
# 10. PARAMETER EFFECT PLOTS
# ============================================================
# These plots help answer:
# - Does learning rate improve F1?
# - Does batch size matter?
# - Which augmentation type performs best?
# - Which optimizer performs best?

def plot_numeric_param_vs_metric(df, param_col, metric_col="f1"):
    """
    Scatter plot of a numeric parameter against a metric.

    Parameters:
        df (pd.DataFrame): Summary dataframe.
        param_col (str): Numeric parameter column.
        metric_col (str): Metric column.
    """
    if param_col not in df.columns or metric_col not in df.columns:
        print(f"[SKIP] Missing column: {param_col} or {metric_col}")
        return

    plot_df = df.dropna(subset=[param_col, metric_col]).copy()

    if len(plot_df) == 0:
        print(f"[SKIP] No data for {param_col} vs {metric_col}")
        return

    plt.figure(figsize=(9, 5))

    for model_name, group_df in plot_df.groupby("model_folder"):
        plt.scatter(group_df[param_col], group_df[metric_col], label=model_name, s=70)

    plt.title(f"{param_col} vs {metric_col.upper()}")
    plt.xlabel(param_col)
    plt.ylabel(metric_col.upper())

    # Learning rate is easier to read on log scale.
    if param_col == "learning_rate":
        plt.xscale("log")

    plt.legend()
    save_current_plot(f"{param_col}_vs_{metric_col}.png")


def plot_categorical_param_vs_metric(df, param_col, metric_col="f1"):
    """
    Bar plot of average/best metric by categorical parameter.

    Parameters:
        df (pd.DataFrame): Summary dataframe.
        param_col (str): Categorical parameter column.
        metric_col (str): Metric column.
    """
    if param_col not in df.columns or metric_col not in df.columns:
        print(f"[SKIP] Missing column: {param_col} or {metric_col}")
        return

    plot_df = df.dropna(subset=[metric_col]).copy()

    if len(plot_df) == 0:
        print(f"[SKIP] No data for {param_col} vs {metric_col}")
        return

    grouped_df = (
        plot_df.groupby(param_col, as_index=False)[metric_col]
        .max()
        .sort_values(by=metric_col, ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(grouped_df[param_col].astype(str), grouped_df[metric_col])
    plt.title(f"Best {metric_col.upper()} by {param_col}")
    plt.xlabel(param_col)
    plt.ylabel(f"Best {metric_col.upper()}")
    plt.xticks(rotation=30, ha="right")
    save_current_plot(f"{param_col}_best_{metric_col}.png")


# Numeric parameters.
for param in ["learning_rate", "batch_size", "epochs", "weight_decay", "num_train_images"]:
    plot_numeric_param_vs_metric(ranked_df, param, "f1")

# Categorical parameters.
for param in ["optimizer", "scheduler", "augmentation_type", "run_type", "runner"]:
    plot_categorical_param_vs_metric(ranked_df, param, "f1")


# ## run plots FOR EVERY MODEL PARAMETER

# In[11]:


# ============================================================
# 11. MODEL-SPECIFIC PARAMETER PLOTS
# ============================================================
# These plots are useful when one model has many experiments.

def plot_model_specific_param_effects(df, model_folder, metric_col="f1"):
    """
    Create model-specific parameter plots.

    Parameters:
        df (pd.DataFrame): Summary dataframe.
        model_folder (str): Model folder to plot.
        metric_col (str): Metric column.
    """
    model_df = df[df["model_folder"] == model_folder].copy()

    if len(model_df) == 0:
        print(f"[SKIP] No experiments for model: {model_folder}")
        return

    print(f"\nCreating parameter plots for: {model_folder}")

    # Learning rate vs metric.
    if "learning_rate" in model_df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(model_df["learning_rate"], model_df[metric_col], s=80)
        plt.xscale("log")
        plt.title(f"{model_folder}: Learning Rate vs {metric_col.upper()}")
        plt.xlabel("learning_rate")
        plt.ylabel(metric_col.upper())
        save_current_plot(f"{model_folder}_learning_rate_vs_{metric_col}.png")

    # Epochs vs metric.
    if "epochs" in model_df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(model_df["epochs"], model_df[metric_col], s=80)
        plt.title(f"{model_folder}: Epochs vs {metric_col.upper()}")
        plt.xlabel("epochs")
        plt.ylabel(metric_col.upper())
        save_current_plot(f"{model_folder}_epochs_vs_{metric_col}.png")

    # Augmentation type.
    if "augmentation_type" in model_df.columns:
        grouped_df = (
            model_df.groupby("augmentation_type", as_index=False)[metric_col]
            .max()
            .sort_values(by=metric_col, ascending=False)
        )

        plt.figure(figsize=(8, 5))
        plt.bar(grouped_df["augmentation_type"].astype(str), grouped_df[metric_col])
        plt.title(f"{model_folder}: Best {metric_col.upper()} by Augmentation")
        plt.xlabel("augmentation_type")
        plt.ylabel(f"Best {metric_col.upper()}")
        plt.xticks(rotation=30, ha="right")
        save_current_plot(f"{model_folder}_augmentation_best_{metric_col}.png")


for model_folder in sorted(ranked_df["model_folder"].dropna().unique()):
    plot_model_specific_param_effects(ranked_df, model_folder, metric_col="f1")


# ## confusion matrices

# In[12]:


# ============================================================
# 12. CONFUSION MATRIX SUMMARY
# ============================================================
# These plots help us see if a model is making more false positives or false negatives.

def plot_error_counts_for_best_per_model(best_df):
    """
    Plot FP and FN counts for best experiment per model.

    Parameters:
        best_df (pd.DataFrame): Best experiment per model dataframe.
    """
    needed_cols = ["model_folder", "fp", "fn"]

    if not all(col in best_df.columns for col in needed_cols):
        print("[SKIP] FP/FN columns missing.")
        return

    plot_df = best_df[needed_cols].copy()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, plot_df["fp"], width, label="False Positives")
    plt.bar(x + width / 2, plot_df["fn"], width, label="False Negatives")

    plt.title("False Positives vs False Negatives for Best Experiment per Model")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(x, plot_df["model_folder"], rotation=30, ha="right")
    plt.legend()

    save_current_plot("fp_fn_best_per_model.png")


if "best_per_model_df" in globals() and len(best_per_model_df) > 0:
    plot_error_counts_for_best_per_model(best_per_model_df)
else:
    print("best_per_model_df not available.")


# # get best model

# ## git helper functions

# In[13]:


# ============================================================
# 13. GIT HELPER FUNCTIONS
# ============================================================
# These functions try to find and extract the exact files from the commit
# where the best experiment was saved.

def run_git_command(args, repo_root=REPO_ROOT, check=False):
    """
    Run a git command and return stdout.

    Parameters:
        args (list): Git command arguments after "git".
        repo_root (Path): Repository root.
        check (bool): Whether to raise error on failure.

    Returns:
        str: Command stdout.
    """
    result = subprocess.run(
        ["git"] + args,
        cwd=repo_root,
        capture_output=True,
        text=True
    )

    if check and result.returncode != 0:
        print("Git command failed:")
        print(" ".join(["git"] + args))
        print(result.stderr)
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def is_git_repo(repo_root=REPO_ROOT):
    """
    Check if this folder is inside a Git repository.

    Returns:
        bool: True if Git repo exists.
    """
    output = run_git_command(["rev-parse", "--is-inside-work-tree"], repo_root=repo_root)
    return output.strip() == "true"


def get_current_git_commit(repo_root=REPO_ROOT):
    """
    Get current HEAD commit hash.

    Returns:
        str or None: Commit hash if available.
    """
    if not is_git_repo(repo_root):
        return None

    commit_hash = run_git_command(["rev-parse", "HEAD"], repo_root=repo_root)
    return commit_hash if commit_hash else None


print(f"Is Git repo: {is_git_repo()}")
print(f"Current commit: {get_current_git_commit()}")


# ## find commit with best model

# In[14]:


# ============================================================
# 14. FIND COMMIT CONTAINING EXPERIMENT ID
# ============================================================
# This searches Git history for the commit where a summary CSV contains the selected experiment_id.

def git_file_exists_at_commit(commit_hash, file_path):
    """
    Check if a file exists at a specific Git commit.

    Parameters:
        commit_hash (str): Commit hash.
        file_path (Path or str): File path relative to repo root.

    Returns:
        bool: True if file exists at commit.
    """
    file_path = str(file_path).replace("\\", "/")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_hash}:{file_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def read_git_file_at_commit(commit_hash, file_path):
    """
    Read a text file from a specific Git commit.

    Parameters:
        commit_hash (str): Commit hash.
        file_path (Path or str): File path relative to repo root.

    Returns:
        str or None: File content if available.
    """
    file_path = str(file_path).replace("\\", "/")

    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{file_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None

    return result.stdout


def find_commit_for_experiment(experiment_id, model_folder):
    """
    Find the latest commit whose model summary CSV contains the experiment_id.

    Parameters:
        experiment_id (str): Experiment ID to search for.
        model_folder (str): Model folder name.

    Returns:
        str or None: Commit hash if found.
    """
    summary_rel_path = Path("outputs") / model_folder / f"{model_folder}_summary.csv"
    summary_rel_path = str(summary_rel_path).replace("\\", "/")

    if not is_git_repo():
        print("Not a Git repo. Cannot search commits.")
        return None

    # Get commits that touched the summary file, newest first.
    log_output = run_git_command([
        "log",
        "--pretty=format:%H",
        "--",
        summary_rel_path
    ])

    commits = [line.strip() for line in log_output.splitlines() if line.strip()]

    if len(commits) == 0:
        print(f"No commits found for summary file: {summary_rel_path}")
        return None

    for commit_hash in commits:
        content = read_git_file_at_commit(commit_hash, summary_rel_path)

        if content is not None and experiment_id in content:
            print(f"Found experiment_id in commit: {commit_hash}")
            return commit_hash

    print(f"Could not find experiment_id in Git history: {experiment_id}")
    return None


if len(ranked_df) > 0:
    best_row = ranked_df.iloc[0].copy()

    BEST_EXPERIMENT_ID = best_row["experiment_id"]
    BEST_MODEL_FOLDER = best_row["model_folder"]

    print("Best experiment selected:")
    print(f"Experiment ID: {BEST_EXPERIMENT_ID}")
    print(f"Model folder:   {BEST_MODEL_FOLDER}")
    print(f"F1:             {best_row.get('f1')}")
    print(f"ROC AUC:        {best_row.get('roc_auc')}")
    print(f"Accuracy:       {best_row.get('accuracy')}")

    BEST_COMMIT_HASH = find_commit_for_experiment(
        experiment_id=BEST_EXPERIMENT_ID,
        model_folder=BEST_MODEL_FOLDER
    )

    print(f"Best commit hash: {BEST_COMMIT_HASH}")
else:
    print("No experiments found.")


# ## extract best model from git commit

# In[15]:


# ============================================================
# 15. EXTRACT BEST MODEL FILES FROM GIT COMMIT
# ============================================================
# This extracts best_model.pth and related run files from the commit
# where the best experiment was found.

def extract_file_from_git_commit(commit_hash, source_rel_path, dest_path):
    """
    Extract a file from a Git commit and save it locally.

    Parameters:
        commit_hash (str): Git commit hash.
        source_rel_path (str): Source file path relative to repo root.
        dest_path (Path): Destination path.

    Returns:
        bool: True if extracted successfully.
    """
    source_rel_path = str(source_rel_path).replace("\\", "/")
    dest_path = Path(dest_path)

    if not git_file_exists_at_commit(commit_hash, source_rel_path):
        print(f"[MISSING IN COMMIT] {source_rel_path}")
        return False

    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{source_rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True
    )

    if result.returncode != 0:
        print(f"[FAILED] Could not extract {source_rel_path}")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as f:
        f.write(result.stdout)

    print(f"[EXTRACTED] {source_rel_path} -> {dest_path}")
    return True


def extract_best_experiment_files(best_row, commit_hash):
    """
    Extract all latest-output files for the best experiment from Git.

    Parameters:
        best_row (pd.Series): Best experiment row.
        commit_hash (str): Commit hash containing that experiment.

    Returns:
        dict: Extraction results.
    """
    if commit_hash is None:
        print("No commit hash available. Cannot extract from Git.")
        return {}

    model_folder = best_row["model_folder"]
    experiment_id = best_row["experiment_id"]

    # Folder where we save the selected best run.
    best_run_dir = BEST_OVERALL_DIR / f"{model_folder}_{experiment_id}"
    best_run_dir.mkdir(parents=True, exist_ok=True)

    # Files expected from our notebook structure.
    files_to_extract = {
        "best_model.pth": Path("outputs") / model_folder / "best_model.pth",
        "config.json": Path("outputs") / model_folder / "config.json",
        "predictions.csv": Path("outputs") / model_folder / "predictions.csv",
        "training_history.csv": Path("outputs") / model_folder / "training_history.csv",
        f"{model_folder}_summary.csv": Path("outputs") / model_folder / f"{model_folder}_summary.csv",
    }

    results = {}

    for output_name, source_path in files_to_extract.items():
        dest_path = best_run_dir / output_name
        success = extract_file_from_git_commit(
            commit_hash=commit_hash,
            source_rel_path=source_path,
            dest_path=dest_path
        )
        results[output_name] = success

    # Save best row metadata separately.
    metadata = {
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_metric": PRIMARY_METRIC,
        "git_commit_hash": commit_hash,
        "best_experiment_row": best_row.to_dict(),
        "extraction_results": results,
        "note": (
            "Files were extracted from Git history. "
            "This is accurate only if the runner committed the model/config/predictions/history "
            "in the same commit as the summary row."
        )
    }

    metadata_path = best_run_dir / "best_experiment_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"\nBest experiment metadata saved to: {metadata_path}")

    return results


if "BEST_COMMIT_HASH" in globals() and BEST_COMMIT_HASH is not None:
    extraction_results = extract_best_experiment_files(best_row, BEST_COMMIT_HASH)
    print("\nExtraction results:")
    print(extraction_results)
else:
    print("No best commit hash found. Skipping Git extraction.")


# ## fallback: if best not found in git, make current = best

# In[16]:


# ============================================================
# 16. FALLBACK: COPY CURRENT FILES IF GIT EXTRACTION FAILS
# ============================================================
# If Git extraction does not work, this copies the current files from the model folder.
# This is useful if the best experiment is also the current/latest run.

def copy_current_best_files(best_row):
    """
    Copy current files from outputs/<model_folder>/ into outputs/best_overall/current_copy/.

    Parameters:
        best_row (pd.Series): Best experiment row.

    Returns:
        Path: Destination folder.
    """
    model_folder = best_row["model_folder"]
    experiment_id = best_row["experiment_id"]

    source_dir = OUTPUTS_DIR / model_folder
    dest_dir = BEST_OVERALL_DIR / f"current_copy_{model_folder}_{experiment_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "best_model.pth",
        "config.json",
        "predictions.csv",
        "training_history.csv",
        f"{model_folder}_summary.csv",
    ]

    copied_files = []

    for filename in files_to_copy:
        source_path = source_dir / filename
        dest_path = dest_dir / filename

        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_files.append(filename)
            print(f"[COPIED] {source_path} -> {dest_path}")
        else:
            print(f"[MISSING CURRENT FILE] {source_path}")

    metadata = {
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "copy_type": "current_files_fallback",
        "best_experiment_row": best_row.to_dict(),
        "copied_files": copied_files,
        "warning": (
            "This copied the current files from the model folder. "
            "Use this only if the best experiment is the current/latest run."
        )
    }

    metadata_path = dest_dir / "current_copy_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"Fallback metadata saved to: {metadata_path}")

    return dest_dir


# Run this manually only if Git extraction failed or you know the best run is the latest run.
# fallback_dir = copy_current_best_files(best_row)


# # save analysis

# In[17]:


# ============================================================
# 17. SAVE FINAL ANALYSIS REPORT TEXT
# ============================================================
# This creates a readable text summary of the current experiment results.

REPORT_PATH = ANALYSIS_DIR / "analysis_report.txt"

with open(REPORT_PATH, "w") as f:
    f.write("Task A Experiment Analysis Report\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total experiments analyzed: {len(ranked_df)}\n")
    f.write(f"Primary ranking metric: {PRIMARY_METRIC}\n\n")

    if len(ranked_df) > 0:
        f.write("Best Overall Experiment\n")
        f.write("-" * 50 + "\n")
        f.write(f"Experiment ID: {best_row.get('experiment_id')}\n")
        f.write(f"Model: {best_row.get('model_name')}\n")
        f.write(f"Model folder: {best_row.get('model_folder')}\n")
        f.write(f"Runner: {best_row.get('runner')}\n")
        f.write(f"F1: {best_row.get('f1')}\n")
        f.write(f"Accuracy: {best_row.get('accuracy')}\n")
        f.write(f"ROC AUC: {best_row.get('roc_auc')}\n")
        f.write(f"Average Precision: {best_row.get('average_precision')}\n")
        f.write(f"Threshold: {best_row.get('threshold')}\n")
        f.write(f"Notes: {best_row.get('notes')}\n\n")

    if "best_per_model_df" in globals() and len(best_per_model_df) > 0:
        f.write("Best Experiment Per Model\n")
        f.write("-" * 50 + "\n")

        for _, row in best_per_model_df.iterrows():
            f.write(
                f"{row.get('model_folder')} | "
                f"F1={row.get('f1')} | "
                f"Acc={row.get('accuracy')} | "
                f"AUC={row.get('roc_auc')} | "
                f"Experiment={row.get('experiment_id')}\n"
            )

print(f"Analysis report saved to: {REPORT_PATH}")


# In[18]:


# ============================================================
# 18. FINAL OUTPUT CHECK
# ============================================================
# Show what the analysis notebook created.

print("Analysis files created:\n")

for path in sorted(ANALYSIS_DIR.rglob("*")):
    if path.is_file():
        print(path.relative_to(REPO_ROOT))

print("\nBest overall files:\n")

for path in sorted(BEST_OVERALL_DIR.rglob("*")):
    if path.is_file():
        print(path.relative_to(REPO_ROOT))

