"""
ALGORITHM 11: MULTI-LAYER PERCEPTRON (MLPClassifier)
---------------------------------------------------
A Multi-Layer Perceptron (MLP) is a feed-forward artificial neural
network composed of fully connected (Dense) layers. Unlike the 1D CNN
implemented in `11_cnn.py`, which learns local feature patterns through
convolutional filters, the MLP treats each transaction as a flat
30-feature input vector and learns non-linear relationships through
stacked hidden layers and backpropagation. The final output layer uses
a sigmoid activation to estimate the probability of a transaction being
fraudulent.

Methodology follows the same leakage-safe workflow used throughout this
project:

  • SMOTE is applied only to the training data within an imbalanced-learn
    Pipeline. This ensures oversampling is performed separately inside
    each cross-validation fold, preventing data leakage.

  • StandardScaler is included inside the Pipeline and fitted only on
    training data. Scaling is particularly important for neural networks,
    as MLP performance is highly sensitive to feature magnitudes.

  • Model evaluation is performed on the original, untouched test set,
    preserving the real-world class imbalance and providing an unbiased
    estimate of generalisation performance.

Computational Considerations:

  • sklearn's MLPClassifier is CPU-based and does not provide the GPU
    acceleration available in modern deep learning frameworks such as
    TensorFlow or PyTorch.

  • To maintain practical training times, hyperparameter optimisation is
    performed on a stratified training subset using a compact search
    space.

  • The best-performing configuration is subsequently retrained on a
    larger SMOTE-balanced training sample before final evaluation on the
    hold-out test set.
"""
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import time
import warnings
import numpy as np
import pandas as pd
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from eval_utils import load_all, evaluate

# MLPClassifier warns loudly on unconverged runs during the search --
# expected here since we cap max_iter to keep the search fast; the
# FINAL model below is trained with a larger max_iter, so it's fine.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()

    # --- Stage 1: small architecture/hyperparameter search on a subsample ---
    # (mirrors the pattern in 10_gradient_boosting.py -- full-data grid
    # search with an MLP would be far too slow on a single CPU core)
    rng = np.random.RandomState(RANDOM_STATE)
    sub_idx = rng.choice(len(X_train), size=min(15000, len(X_train)), replace=False)
    X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),           # MLPs need scaled inputs; trees didn't
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", MLPClassifier(
            random_state=RANDOM_STATE,
            early_stopping=True,       # holds out part of the (balanced) train fold
            n_iter_no_change=10,
            validation_fraction=0.1,
            max_iter=150,              # kept modest during the search for speed
        )),
    ])

    # Vary depth/width, activation, regularization strength (alpha) and
    # learning rate -- the "usual suspects" for tuning a plain MLP.
    param_dist = {
        "clf__hidden_layer_sizes": [(64, 32), (128, 64), (100, 50, 25)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-4],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.time()
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=6, scoring="average_precision",
        cv=cv, n_jobs=1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_sub, y_sub)
    print(f"Best params: {search.best_params_} (search took {time.time() - t0:.1f}s)")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")

    # --- Stage 2: refit the winning config on a larger SMOTE-balanced sample ---
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

    fit_idx = rng.choice(len(X_train), size=min(60000, len(X_train)), replace=False)
    X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = sm.fit_resample(X_fit_s, y_fit)

    t0 = time.time()
    best_mlp = MLPClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        max_iter=400,          # generous budget for the real final fit
    )
    best_mlp.fit(X_bal, y_bal)
    print(f"Refit on {len(X_bal)}-row SMOTE-balanced sample took {time.time() - t0:.1f}s "
          f"({best_mlp.n_iter_} iterations, converged={best_mlp.n_iter_ < best_mlp.max_iter})")

    # --- Evaluate on the full, untouched, real-imbalance test set ---
    y_pred = best_mlp.predict(X_test_s)
    y_score = best_mlp.predict_proba(X_test_s)[:, 1]
    result = evaluate("MLP (Multi-Layer Perceptron, tuned)", y_test, y_pred, y_score,
                       note=str(best_params))

    pd.DataFrame([result]).to_csv("outputs/results/results_mlp.csv", index=False)

    # Save both the model AND the scaler it expects its inputs to be in --
    # unlike the tree models, this one is useless without matching scaling.
    joblib.dump(best_mlp, "outputs/models/model_mlp.joblib")
    joblib.dump(scaler, "outputs/models/scaler_mlp.joblib")
    print("\nSaved outputs/results/results_mlp.csv, outputs/models/model_mlp.joblib "
          "and outputs/models/scaler_mlp.joblib")
