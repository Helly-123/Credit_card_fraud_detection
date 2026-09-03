"""
ALGORITHM 9: DIMENSIONALITY REDUCTION (PCA)
------------------------------------------------------
The Credit Card Fraud Detection dataset is unusual because features
V1-V28 are already Principal Component Analysis (PCA) components
generated during the original anonymization process. Consequently,
this experiment investigates the effects of applying an additional
round of dimensionality reduction rather than performing PCA for the
first time.

The objective is not necessarily to improve predictive performance,
but to evaluate the trade-off between model complexity, training time,
and fraud-detection effectiveness as feature dimensionality decreases.

To isolate the impact of dimensionality reduction, a fixed classifier
(Logistic Regression) is used across all experiments. Logistic
Regression was chosen because it is computationally efficient,
interpretable, and highly sensitive to changes in feature-space
representation, making it well suited for comparing PCA configurations.

The following feature configurations are evaluated:

  1. All 30 original features (baseline).
  2. PCA retaining enough components to explain 95% of total variance.
  3. PCA reduced to 10 components.
  4. PCA reduced to 5 components.

For each configuration:
  - Features are standardized using StandardScaler.
  - SMOTE is applied only to the training data to address class
    imbalance.
  - Logistic Regression is trained on the transformed feature space.
  - Evaluation is performed on the full, untouched test set.

The experiment measures whether reducing dimensionality can:
  - Decrease computational cost and training time.
  - Reduce noise and redundant information.
  - Maintain fraud-detection performance with fewer features.

Key Evaluation Metrics:
  - F1 Score (Fraud Class)
  - Precision
  - Recall
  - ROC-AUC
  - PR-AUC
  - Model Training Time

Expected Outcome:
  If performance remains similar after substantial dimensionality
  reduction, this suggests that most predictive information is
  concentrated within a relatively small number of components.
  Conversely, a significant decline in performance would indicate
  that important fraud-related information is distributed across
  a larger portion of the feature space.

This analysis provides insight into the relationship between feature
dimensionality, computational efficiency, and predictive performance,
which is a central consideration in real-world fraud detection systems.
"""
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import time
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from eval_utils import load_all, evaluate

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()
    results = []

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Find how many components explain 95% variance
    pca_full = PCA(random_state=42).fit(X_train_s)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)
    print(f"Components needed for 95% variance: {n_95} (out of {X_train.shape[1]})")

    configs = [
        ("All 30 features (no reduction)", None),
        (f"PCA -> {n_95} components (95% variance)", n_95),
        ("PCA -> 10 components", 10),
        ("PCA -> 5 components", 5),
    ]

    for label, n_comp in configs:
        if n_comp is None:
            Xtr, Xte = X_train_s, X_test_s
        else:
            pca = PCA(n_components=n_comp, random_state=42)
            Xtr = pca.fit_transform(X_train_s)
            Xte = pca.transform(X_test_s)

        sm = SMOTE(random_state=42)
        Xtr_bal, ytr_bal = sm.fit_resample(Xtr, y_train)

        t0 = time.time()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr_bal, ytr_bal)
        fit_time = time.time() - t0

        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]
        res = evaluate(f"LogReg + {label}", y_test, y_pred, y_score,
                        note=f"fit_time={fit_time:.2f}s, n_features={Xtr.shape[1]}")
        res["fit_time_sec"] = fit_time
        res["n_features"] = Xtr.shape[1]
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("outputs/results/results_dim_reduction.csv", index=False)
    print("\n=== Dimensionality Reduction trade-off summary ===")
    print(df[["model", "f1_fraud", "pr_auc", "fit_time_sec", "n_features"]].to_string(index=False))
    print("\nSaved outputs/results/results_dim_reduction.csv")
