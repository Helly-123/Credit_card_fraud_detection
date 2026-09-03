"""
Exploratory Data Analysis (EDA)
---------------------------------------
Generated plots:
1. Class Distribution
2. Feature Correlation with Fraud Class

Outputs:
outputs/plots/01_class_distribution.png
outputs/plots/02_feature_correlation.png
"""

# --------------------------------------------------
# Project Path Setup
# --------------------------------------------------

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)

for folder in ["plots", "models", "results", "interim"]:
    os.makedirs(f"outputs/{folder}", exist_ok=True)

# --------------------------------------------------
# Imports
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import import_module

# Reuse loader from Step 1
load_data = import_module("01_load_data").load_data

# --------------------------------------------------
# Plot Style
# --------------------------------------------------

sns.set_theme(style="whitegrid", palette="deep")

LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#C44E52"


# --------------------------------------------------
# EDA Function
# --------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """
    Generates and saves EDA plots.
    """

    # ==================================================
    # Plot 1: Class Distribution
    # ==================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Class",
        hue="Class",
        data=df,
        palette=[LEGIT_COLOR, FRAUD_COLOR],
        legend=False,
        ax=ax,
    )

    ax.set_title("Class Distribution")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (
                p.get_x() + p.get_width() / 2,
                p.get_height(),
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/01_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # Plot 2: Feature Correlation with Fraud Class
    # ==================================================

    corr_with_class = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 10))

    corr_with_class.plot(
        kind="barh",
        color=[
            FRAUD_COLOR if x > 0 else LEGIT_COLOR
            for x in corr_with_class
        ],
        ax=ax,
    )

    ax.set_title("Feature Correlation with Fraud Class")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/02_feature_correlation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # Summary
    # ==================================================

    print("\nTop 5 Positive Fraud Correlations")
    print(corr_with_class.tail(5))

    print("\nTop 5 Negative Fraud Correlations")
    print(corr_with_class.head(5))

    print("\nEDA completed successfully.")
    print("Saved plots:")
    print(" - outputs/plots/01_class_distribution.png")
    print(" - outputs/plots/02_feature_correlation.png")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    run_eda(df)