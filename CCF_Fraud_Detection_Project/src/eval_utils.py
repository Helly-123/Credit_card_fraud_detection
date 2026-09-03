"""
Shared helper: load original (imbalanced) train/test splits, and a
standard evaluate() function so every algorithm gets scored identically
on the SAME untouched test set with the SAME metrics.
"""
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score, accuracy_score
)


def load_all():
    X_train = pd.read_parquet("outputs/interim/X_train.parquet")
    y_train = pd.read_parquet("outputs/interim/y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet("outputs/interim/X_test.parquet")
    y_test = pd.read_parquet("outputs/interim/y_test.parquet").iloc[:, 0]
    return X_train, y_train, X_test, y_test


def evaluate(name, y_test, y_pred, y_score=None, note=""):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")
    pr_auc = average_precision_score(y_test, y_score) if y_score is not None else float("nan")

    print(f"\n=== {name} === {note}")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Legit", "Fraud"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {acc:.4f} | Precision(fraud): {prec:.4f} | Recall(fraud): {rec:.4f} | "
          f"F1(fraud): {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {
        "model": name, "accuracy": acc, "precision_fraud": prec, "recall_fraud": rec,
        "f1_fraud": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "note": note,
    }
