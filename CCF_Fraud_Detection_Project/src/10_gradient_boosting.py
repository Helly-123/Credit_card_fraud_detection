"""
ALGORITHM 10: GRADIENT BOOSTING CLASSIFIER (sklearn)
------------------------------------------------------
This experiment evaluates the classical Gradient Boosting Classifier
implemented in scikit-learn. Unlike XGBoost, LightGBM, or CatBoost,
which are highly optimized gradient boosting frameworks, sklearn's
GradientBoostingClassifier represents the original boosting approach
and serves as a useful baseline for comparison.

Gradient Boosting builds an ensemble of weak decision trees
sequentially. Each new tree is trained to correct the residual errors
made by the previous trees, gradually improving overall predictive
performance. By combining many weak learners, the model can capture
complex non-linear relationships within the data.

For credit card fraud detection, Gradient Boosting is capable of
learning subtle interactions among the PCA-transformed transaction
features and identifying patterns associated with fraudulent behavior.

Training Procedure:
  1. A stratified random subsample of the training data is used for
     hyperparameter optimization to reduce computational cost.
  2. SMOTE is applied only to the training folds inside the pipeline,
     ensuring leakage-free cross-validation.
  3. RandomizedSearchCV is used to tune key boosting parameters.
  4. The best configuration is refitted on a larger SMOTE-balanced
     training sample.
  5. Performance is evaluated on the full, untouched test set.

Key Hyperparameters:
  - n_estimators:
      Number of boosting stages (decision trees).
      More trees generally improve learning capacity but increase
      training time and risk of overfitting.

  - max_depth:
      Maximum depth of individual trees.
      Controls model complexity and feature interaction strength.

  - learning_rate:
      Shrinks the contribution of each tree.
      Smaller values often improve generalization but require more trees.

Evaluation Metrics:
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - PR-AUC (most important for class-imbalanced fraud detection)

Resource Considerations:
  sklearn's GradientBoostingClassifier lacks many of the computational
  optimizations available in modern boosting libraries, such as
  histogram-based tree construction, advanced parallelization, and
  highly efficient memory management. Consequently, training is
  significantly slower on large datasets.

  To maintain a fair comparison while keeping runtime practical,
  hyperparameter tuning is performed on a smaller stratified sample,
  followed by final training on a larger SMOTE-balanced subset rather
  than the full balanced dataset.

Research Objective:
  This experiment investigates how a traditional gradient boosting
  approach compares with modern ensemble methods (XGBoost, LightGBM,
  CatBoost), classical machine learning algorithms, and deep learning
  models in terms of fraud-detection performance, computational cost,
  and robustness to severe class imbalance.

Expected Outcome:
  Gradient Boosting is expected to outperform simpler models such as
  Logistic Regression while generally achieving lower predictive
  performance and longer training times than modern boosting
  implementations. The experiment therefore provides an important
  historical and methodological benchmark within the overall study.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from eval_utils import load_all, evaluate

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()

    # sklearn's GradientBoostingClassifier has no early stopping / histogram
    # optimization, so it is very slow on a single CPU core. We keep the
    # search small and cap n_estimators, and skip refitting on the full
    # ~450k-row balanced set (that alone would take much longer here) --
    # we fit the final model directly on a 50k-row SMOTE-balanced sample
    # instead, which is still a fair, honest comparison point.
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train), size=min(15000, len(X_train)), replace=False)
    X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]

    pipe = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", GradientBoostingClassifier(random_state=42, validation_fraction=0.1,
                                            n_iter_no_change=5, tol=1e-4)),
    ])
    param_dist = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    t0 = time.time()
    search = RandomizedSearchCV(pipe, param_dist, n_iter=4, scoring="average_precision",
                                 cv=cv, n_jobs=1, random_state=42, verbose=1)
    search.fit(X_sub, y_sub)
    print(f"Best params: {search.best_params_} (search took {time.time()-t0:.1f}s)")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")

    # Refit best params on a larger (but still capped) SMOTE-balanced sample
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    fit_idx = rng.choice(len(X_train), size=min(60000, len(X_train)), replace=False)
    sm = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

    t0 = time.time()
    best_gb = GradientBoostingClassifier(**best_params, random_state=42,
                                          validation_fraction=0.1, n_iter_no_change=5, tol=1e-4)
    best_gb.fit(X_tr_bal, y_tr_bal)
    print(f"Refit on 60k-row sample took {time.time()-t0:.1f}s")

    y_pred = best_gb.predict(X_test)
    y_score = best_gb.predict_proba(X_test)[:, 1]
    result = evaluate("Gradient Boosting (tuned)", y_test, y_pred, y_score, note=str(best_params))

    pd.DataFrame([result]).to_csv("outputs/results/results_gradient_boosting.csv", index=False)
    print("\nSaved outputs/results/results_gradient_boosting.csv")
