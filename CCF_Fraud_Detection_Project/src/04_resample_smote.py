"""
Handling Class Imbalance
-----------------------------------------------------------------------------
Credit card fraud detection is a highly imbalanced classification problem.
In the original dataset, legitimate transactions vastly outnumber fraudulent
transactions:

    Legitimate Transactions : 284,315
    Fraudulent Transactions : 492

Fraud cases account for approximately 0.17% of all observations. Under such
conditions, a model may achieve very high accuracy by predominantly predicting
the majority class while failing to identify fraudulent transactions
effectively. Therefore, accuracy alone is not a reliable performance metric
for this task.

To mitigate the effects of class imbalance, resampling techniques are applied
during model development.

IMPORTANT PRINCIPLE
-------------------
Resampling is performed exclusively on the training data.

The following practices must be avoided:

    • Applying resampling before the train/test split
    • Applying resampling to the test set

The test set must preserve the original class distribution to ensure that
model evaluation reflects real-world operating conditions and provides an
unbiased estimate of performance on unseen data.

Resampling Strategies Evaluated
-------------------------------

1. Original Training Data
   No class balancing is applied. Models are trained on the naturally
   imbalanced dataset.

2. SMOTE (Synthetic Minority Over-sampling Technique)
   Generates synthetic minority-class examples by interpolating between
   existing fraud samples, thereby increasing minority-class representation
   without simply duplicating observations.

3. Random Under-Sampling (RUS)
   Randomly removes majority-class examples to create a more balanced
   training dataset. While computationally efficient, this approach may
   discard potentially useful information from legitimate transactions.
-----------------------------------------------------------------------------
"""
# ==================================================
# PROJECT PATH SETUP
# ==================================================

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)

for folder in ["plots", "models", "results", "interim"]:
    os.makedirs(f"outputs/{folder}", exist_ok=True)

# ==================================================
# IMPORTS
# ==================================================

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ==================================================
# LOAD TRAINING DATA
# ==================================================

def load_training_data():
    """
    Load training data generated in Step 3.
    """

    X_train = pd.read_parquet(
        "outputs/interim/X_train.parquet"
    )

    y_train = pd.read_parquet(
        "outputs/interim/y_train.parquet"
    ).iloc[:, 0]

    return X_train, y_train


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    X_train, y_train = load_training_data()

    print("\n" + "=" * 60)
    print("ORIGINAL TRAINING SET")
    print("=" * 60)

    print(y_train.value_counts())

    # ==================================================
    # 1. ORIGINAL TRAINING DATA
    # ==================================================

    print("\nSaving Original Training Dataset...")

    X_train.to_parquet(
        "outputs/interim/X_train_original.parquet"
    )

    y_train.to_frame().to_parquet(
        "outputs/interim/y_train_original.parquet"
    )

    # ==================================================
    # 2. SMOTE
    # ==================================================

    print("\n" + "=" * 60)
    print("APPLYING SMOTE")
    print("=" * 60)

    smote = SMOTE(
        random_state=42
    )

    X_smote, y_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print(y_smote.value_counts())

    X_smote.to_parquet(
        "outputs/interim/X_train_smote.parquet"
    )

    y_smote.to_frame().to_parquet(
        "outputs/interim/y_train_smote.parquet"
    )

    # ==================================================
    # 3. RANDOM UNDERSAMPLING
    # ==================================================

    print("\n" + "=" * 60)
    print("APPLYING RANDOM UNDERSAMPLING")
    print("=" * 60)

    rus = RandomUnderSampler(
        random_state=42
    )

    X_rus, y_rus = rus.fit_resample(
        X_train,
        y_train
    )

    print(y_rus.value_counts())

    X_rus.to_parquet(
        "outputs/interim/X_train_rus.parquet"
    )

    y_rus.to_frame().to_parquet(
        "outputs/interim/y_train_rus.parquet"
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print("\n" + "=" * 60)
    print("RESAMPLING COMPLETE")
    print("=" * 60)

    print("\nSaved Datasets:")

    print("\n1. Original Dataset")
    print("   X_train_original.parquet")
    print("   y_train_original.parquet")

    print("\n2. SMOTE Dataset")
    print("   X_train_smote.parquet")
    print("   y_train_smote.parquet")

    print("\n3. Random UnderSampling Dataset")
    print("   X_train_rus.parquet")
    print("   y_train_rus.parquet")

    print("\nTest set remains untouched.")
    print("No data leakage introduced.")
