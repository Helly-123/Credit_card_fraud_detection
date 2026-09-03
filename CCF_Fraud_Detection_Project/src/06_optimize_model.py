"""
STEP 6: Hyperparameter Optimisation of the Best-Performing Model (XGBoost)
=============================================================================

Purpose
-------
Based on the results obtained during the comparative model evaluation
stage, XGBoost was identified as the strongest-performing algorithm for
credit card fraud detection. This stage focuses on optimising the model's
hyperparameters to further improve predictive performance and
generalisation capability.

Methodology
-----------
Hyperparameter tuning is performed using RandomizedSearchCV, which
efficiently explores a predefined search space by evaluating a randomly
selected subset of parameter combinations. Compared with exhaustive grid
search, this approach substantially reduces computational requirements
while still identifying high-performing configurations.

A Stratified K-Fold cross-validation strategy is employed during the
optimisation process. Stratification ensures that each validation fold
preserves a similar proportion of fraudulent and legitimate transactions,
thereby providing a more reliable estimate of model performance on
imbalanced data.

Performance Metric
------------------
Average Precision (PR-AUC) is used as the optimisation objective rather
than classification accuracy. This decision is particularly important in
fraud detection because fraudulent transactions represent only a very
small percentage of all observations. Under such conditions, accuracy can
be misleading, whereas PR-AUC provides a more informative assessment of a
model's ability to identify the minority fraud class.

Important Considerations
------------------------
Hyperparameter optimisation is conducted exclusively on the SMOTE-
resampled training dataset. The independent test set remains completely
unseen throughout the tuning process and is used only once for final model
evaluation. This approach prevents data leakage and ensures an unbiased
assessment of real-world performance.

To improve computational efficiency, a random subset of the balanced
training data is used during the hyperparameter search stage. Once the
optimal parameter configuration is identified, the final XGBoost model is
retrained using the complete SMOTE-resampled training dataset before
being evaluated on the original test set.

Outputs Generated
-----------------
This script produces:

    • Optimised XGBoost hyperparameters
    • Cross-validation PR-AUC results
    • Final test-set evaluation metrics
    • Classification report and confusion matrix
    • Optimised model artifact (.joblib)

The final optimised model is saved for subsequent explainability
analysis, model interpretation, and deployment experimentation.

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
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
import joblib

def load_all():
    X_train = pd.read_parquet("outputs/interim/X_train_smote.parquet")
    y_train = pd.read_parquet("outputs/interim/y_train_smote.parquet").iloc[:, 0]
    X_test = pd.read_parquet("outputs/interim/X_test.parquet")
    y_test = pd.read_parquet("outputs/interim/y_test.parquet").iloc[:, 0]
    return X_train, y_train, X_test, y_test


PARAM_DIST = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()

    # Subsample the (already-balanced) SMOTE training set for search speed;
    # final model is refit on the full set with the best params found.
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_train), size=60000, replace=False)
    X_search = X_train.iloc[sample_idx]
    y_search = y_train.iloc[sample_idx]

    base_model = XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_search, y_search)

    print("Best params:", search.best_params_)
    print("Best CV PR-AUC:", search.best_score_)

    # Refit best-found params on the FULL SMOTE training set
    best_model = XGBClassifier(**search.best_params_, eval_metric="logloss", n_jobs=-1, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Optimized XGBoost - Test set performance ===")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Legit", "Fraud"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f} | PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

    joblib.dump(best_model, "outputs/models/model_xgboost_optimized.joblib")
    print("\nSaved optimized model to outputs/models/model_xgboost_optimized.joblib")
