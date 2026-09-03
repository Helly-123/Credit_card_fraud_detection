# -----------------------------------------------------------------------------
# Train/Test Split and Feature Engineering (Feature Scaling)
# -----------------------------------------------------------------------------
# To ensure a robust and unbiased evaluation, the dataset is split into
# training and testing subsets before any resampling or preprocessing steps.
# This approach prevents information leakage from the test set into the
# training process, which could otherwise lead to overly optimistic
# performance estimates.
#
# Feature scaling is performed using parameters learned exclusively from the
# training data and subsequently applied to both the training and test sets.
# This follows standard machine learning practice and preserves the integrity
# of the evaluation process.
#
# The variables V1–V28 are anonymized PCA-transformed features that are
# already approximately standardized, with values centered around zero and
# exhibiting similar variance. Consequently, additional scaling of these
# features is unnecessary. In contrast, the Time and Amount attributes exist
# on substantially different scales and are therefore standardized to ensure
# comparable feature magnitudes during model training.
# -----------------------------------------------------------------------------
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path
from importlib import import_module

load_data = import_module("01_load_data").load_data

Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)
Path("outputs/results").mkdir(parents=True, exist_ok=True)
Path("outputs/interim").mkdir(parents=True, exist_ok=True)

def split_and_scale(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratify to preserve the ~0.17% fraud ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale Time & Amount only. RobustScaler is preferred over StandardScaler
    # here because Amount is heavily right-skewed with extreme outliers.
    scaler = RobustScaler()
    cols_to_scale = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    print("Train shape:", X_train.shape, "| Fraud rate:", y_train.mean())
    print("Test shape :", X_test.shape, "| Fraud rate:", y_test.mean())

    # Persist artifacts so later steps don't need to redo this work
    X_train.to_parquet("outputs/interim/X_train.parquet")
    X_test.to_parquet("outputs/interim/X_test.parquet")
    y_train.to_frame().to_parquet("outputs/interim/y_train.parquet")
    y_test.to_frame().to_parquet("outputs/interim/y_test.parquet")
    joblib.dump(scaler, "outputs/interim/scaler.joblib")
    print("\nSaved train/test splits + scaler to outputs/")
