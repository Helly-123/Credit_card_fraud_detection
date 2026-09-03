"""
STEP 5: Machine Learning Model Training and Comparative Evaluation
=============================================================================

Purpose
-------
This stage trains and compares multiple machine learning algorithms for
credit card fraud detection using the preprocessed datasets generated in
previous stages. The objective is to identify the model that achieves the
best balance between detecting fraudulent transactions and minimizing
false alarms.

Methodology
-----------
To address the severe class imbalance inherent in fraud detection data,
models are trained using the SMOTE-resampled training dataset. However,
evaluation is performed exclusively on the original, untouched test set.
Maintaining the natural class distribution within the test data provides
an unbiased estimate of model performance under realistic operational
conditions and prevents misleading evaluation results.

Models Evaluated
----------------
The following machine learning algorithms are included in the comparative
analysis:

    • Logistic Regression
    • Decision Tree
    • Random Forest
    • XGBoost

Evaluation Metrics
------------------
Model performance is assessed using multiple classification metrics,
including:

    • Precision
    • Recall
    • F1-Score
    • ROC-AUC
    • PR-AUC (Precision-Recall Area Under the Curve)

Particular emphasis is placed on PR-AUC because it is more informative
than accuracy when evaluating highly imbalanced datasets such as credit
card fraud transactions.

Outputs Generated
-----------------
This script produces:

    • Trained model artifacts (.joblib)
    • Classification reports
    • Confusion matrices
    • Model comparison metrics
    • Performance summary tables

=============================================================================
"""

import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)



import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  # LinearSVC instead of SVC: full SVC is too slow on 450k rows
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score
)
import joblib
from pathlib import Path

Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)
Path("outputs/results").mkdir(parents=True, exist_ok=True)
Path("outputs/interim").mkdir(parents=True, exist_ok=True)

def load_all():
    X_train = pd.read_parquet("outputs/interim/X_train_smote.parquet")
    y_train = pd.read_parquet("outputs/interim/y_train_smote.parquet").iloc[:, 0]
    X_test = pd.read_parquet("outputs/interim/X_test.parquet")
    y_test = pd.read_parquet("outputs/interim/y_test.parquet").iloc[:, 0]
    return X_train, y_train, X_test, y_test


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss", n_jobs=-1, random_state=42),
}


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)  # more informative than ROC-AUC for rare events

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Legit", "Fraud"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1(fraud): {f1:.4f}")

    return {"model": name, "f1_fraud": f1, "roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()
    results = []

    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        res = evaluate(name, model, X_test, y_test)
        res["train_time_sec"] = round(train_time, 1)
        results.append(res)
        joblib.dump(model, f"outputs/models/model_{name.replace(' ', '_').lower()}.joblib")

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n\n=== SUMMARY (sorted by PR-AUC, the metric that matters most here) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("outputs/results/model_comparison.csv", index=False)
