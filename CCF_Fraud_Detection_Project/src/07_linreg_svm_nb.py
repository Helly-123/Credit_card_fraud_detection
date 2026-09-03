"""
ALGORITHMS: Linear Regression, Support Vector Machine, and Naive Bayes
=============================================================================

Purpose
-------
This stage implements and evaluates three additional algorithms identified
in the literature for credit card fraud detection:

    • Linear Regression
    • Support Vector Machine (SVM)
    • Gaussian Naive Bayes

The objective is to provide a comprehensive comparison of traditional
machine learning approaches and assess their effectiveness in detecting
rare fraudulent transactions.

Cross-Validation and Resampling Strategy
----------------------------------------
A key methodological consideration in fraud detection is the prevention
of data leakage during model development. To ensure a valid evaluation
process, resampling is incorporated within an imbalanced-learn Pipeline,
allowing SMOTE to be applied independently inside each training fold of
cross-validation.

Hyperparameter optimisation is performed using RandomizedSearchCV in
combination with Stratified K-Fold cross-validation. This approach
ensures that class proportions remain consistent across validation folds,
while preventing synthetic samples from influencing validation data.

Importantly, SMOTE is never applied before cross-validation and never
applied to the test set. This prevents overly optimistic performance
estimates and provides a more realistic assessment of model behaviour on
previously unseen data.

Model Training and Evaluation
-----------------------------
Once the optimal hyperparameters have been identified, each algorithm is
retrained using the full SMOTE-resampled training dataset. Final model
evaluation is then conducted on the original, untouched test dataset,
which preserves the natural class imbalance observed in real-world credit
card transactions.

Evaluation Metrics
------------------
Performance is assessed using metrics appropriate for highly imbalanced
classification tasks, including:

    • Precision
    • Recall
    • F1-Score
    • ROC-AUC
    • PR-AUC (Average Precision)

Particular emphasis is placed on PR-AUC because it provides a more
informative assessment of minority-class detection performance than
classification accuracy in rare-event fraud detection scenarios.


To enable binary classification, predicted outputs are treated as
continuous fraud scores and converted into class predictions using an
optimised decision threshold determined exclusively from training data.
The test set is never used during threshold selection, thereby preserving
evaluation integrity.

Outputs Generated
-----------------
This script produces:

    • Tuned model configurations
    • Fraud detection performance metrics
    • Classification reports
    • Comparative evaluation results
    • CSV summary outputs for subsequent analysis

=============================================================================
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
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from eval_utils import load_all, evaluate

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    # ============================================================
    # LINEAR REGRESSION
    # ============================================================
    print("\n" + "=" * 60 + "\n1) LINEAR REGRESSION\n" + "=" * 60)
    sm = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_tr_bal, y_tr_bal)

    # Find best threshold using training predictions (not test!)
    train_scores = lr.predict(X_tr_bal)
    thresholds = np.linspace(train_scores.min(), train_scores.max(), 200)
    from sklearn.metrics import f1_score as f1s
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1_t = f1s(y_tr_bal, (train_scores >= t).astype(int))
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t
    print(f"Chosen decision threshold (from training data): {best_t:.4f}")

    test_scores = lr.predict(X_test)
    y_pred = (test_scores >= best_t).astype(int)
    results.append(evaluate("Linear Regression (thresholded)", y_test, y_pred, test_scores,
                             note=f"threshold={best_t:.3f}, chosen on train data only"))

    # ============================================================
    # SVM (Linear kernel -- RBF SVC does not scale to this dataset size
    # even after subsetting; LinearSVC is the practical choice)
    # ============================================================
    print("\n" + "=" * 60 + "\n4) SVM (LinearSVC)\n" + "=" * 60)
    svm_pipe = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", LinearSVC(max_iter=5000, random_state=42)),
    ])
    svm_param_dist = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}

    # Search on a stratified subsample for CPU-time reasons; final model
    # is refit on the FULL SMOTE-resampled training set below.
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train), size=min(40000, len(X_train)), replace=False)
    X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]

    t0 = time.time()
    svm_search = RandomizedSearchCV(svm_pipe, svm_param_dist, n_iter=5, scoring="average_precision",
                                     cv=cv, n_jobs=1, random_state=42, verbose=1)
    svm_search.fit(X_sub, y_sub)
    print(f"Best SVM params: {svm_search.best_params_} (search took {time.time()-t0:.1f}s)")

    best_svm = LinearSVC(C=svm_search.best_params_["clf__C"], max_iter=5000, random_state=42)
    best_svm.fit(X_tr_bal, y_tr_bal)
    y_pred = best_svm.predict(X_test)
    y_score = best_svm.decision_function(X_test)
    results.append(evaluate("SVM (LinearSVC, tuned C)", y_test, y_pred, y_score,
                             note=f"C={svm_search.best_params_['clf__C']}"))

    # ============================================================
    # NAIVE BAYES
    # ============================================================
    print("\n" + "=" * 60 + "\n5) NAIVE BAYES\n" + "=" * 60)
    nb_pipe = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", GaussianNB()),
    ])
    nb_param_dist = {"clf__var_smoothing": np.logspace(-12, -6, 20)}
    nb_search = RandomizedSearchCV(nb_pipe, nb_param_dist, n_iter=10, scoring="average_precision",
                                    cv=cv, n_jobs=-1, random_state=42, verbose=1)
    nb_search.fit(X_train, y_train)
    print(f"Best Naive Bayes params: {nb_search.best_params_}")

    best_nb = GaussianNB(var_smoothing=nb_search.best_params_["clf__var_smoothing"])
    best_nb.fit(X_tr_bal, y_tr_bal)
    y_pred = best_nb.predict(X_test)
    y_score = best_nb.predict_proba(X_test)[:, 1]
    results.append(evaluate("Naive Bayes (tuned var_smoothing)", y_test, y_pred, y_score,
                             note=f"var_smoothing={nb_search.best_params_['clf__var_smoothing']:.2e}"))

    pd.DataFrame(results).to_csv("outputs/results/results_batch1_lr_svm_nb.csv", index=False)
    print("\nSaved outputs/results/results_batch1_lr_svm_nb.csv")
