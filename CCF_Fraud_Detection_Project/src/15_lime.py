"""
STEP 15: LIME (Local Interpretable Model-Agnostic Explanations)
-----------------------------------------------------------------------------
Explains the same model and same three transactions as 14_shap.py (loaded
from outputs/interim/xai_example_indices.json).

Produces:
  - Local explanations for the same true positive / false positive / false negative
  - A stability check (re-explain one instance 10x, unseeded, see if the
    top feature changes - LIME uses random perturbation, SHAP doesn't)
  - A cost timing, compared against SHAP's from 14_shap.py
"""
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import json
import time

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from eval_utils import load_all

RANDOM_STATE = 42
MODEL_PATH = "outputs/models/model_xgboost.joblib"
INDICES_PATH = "outputs/interim/xai_example_indices.json"
SHAP_COST_PATH = "outputs/results/shap_cost_and_fidelity.json"

N_LIME_FEATURES = 15
N_STABILITY_RUNS = 10
COST_N_INSTANCES = 100  # must match 14_shap.py


def load_example_indices():
    if not os.path.exists(INDICES_PATH):
        raise FileNotFoundError(f"{INDICES_PATH} not found. Run 14_shap.py first.")
    with open(INDICES_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    X_train, y_train, X_test, y_test = load_all()

    examples = load_example_indices()["examples"]

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["Legit", "Fraud"],
        mode="classification",
        random_state=RANDOM_STATE,
    )

    # ---- Local explanations ----
    print("\n" + "=" * 60)
    print("LOCAL EXPLANATIONS")
    print("=" * 60)

    for case_name, idx in examples.items():
        if idx is None:
            print(f"[warning] No {case_name} example saved by 14_shap.py; skipping.")
            continue

        row = X_test.loc[idx].values
        pred_proba = model.predict_proba(X_test.loc[[idx]])[0, 1]

        print(f"\n--- {case_name} (test row index {idx}) ---")
        print(f"True label: {int(y_test.loc[idx])} | Fraud probability: {pred_proba:.4f}")

        exp = explainer.explain_instance(row, model.predict_proba, num_features=N_LIME_FEATURES)
        print("Top LIME features:")
        for feat, weight in exp.as_list()[:5]:
            print(f"    {feat:35s} {weight:+.4f}")

        out_path = f"outputs/plots/lime_{case_name}.html"
        exp.save_to_file(out_path)
        print(f"Saved {out_path}")

    # ---- Stability check (RQ3) ----
    print("\n" + "=" * 60)
    print(f"STABILITY: re-explaining the same instance {N_STABILITY_RUNS} times")
    print("=" * 60)

    stability_target = examples.get("true_positive") or next(
        (idx for idx in examples.values() if idx is not None), None
    )

    if stability_target is not None:
        row = X_test.loc[stability_target].values
        top_feature_per_run = []
        stability_rows = []

        for run in range(N_STABILITY_RUNS):
            # fresh, unseeded explainer each run -- seeding would defeat the point
            unseeded = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["Legit", "Fraud"],
                mode="classification",
            )
            exp = unseeded.explain_instance(row, model.predict_proba, num_features=N_LIME_FEATURES)
            top_feature, top_weight = exp.as_list()[0]
            top_feature_per_run.append(top_feature)
            stability_rows.append({"run": run, "top_feature": top_feature, "top_feature_weight": top_weight})
            print(f"  run {run}: top feature = {top_feature}")

        n_unique = len(set(top_feature_per_run))
        most_common = max(set(top_feature_per_run), key=top_feature_per_run.count)
        agreement = top_feature_per_run.count(most_common) / N_STABILITY_RUNS
        print(f"\n{n_unique} distinct top feature(s) across {N_STABILITY_RUNS} runs "
              f"(1 = fully stable). Most common appeared in {agreement:.0%} of runs.")

        pd.DataFrame(stability_rows).to_csv("outputs/results/lime_stability.csv", index=False)
        print("Saved outputs/results/lime_stability.csv")
    else:
        print("[warning] No example available for the stability check.")

    # ---- Cost timing + comparison  ----
    print("\n" + "=" * 60)
    print(f"COST: explaining {COST_N_INSTANCES} instances with LIME")
    print("=" * 60)

    X_cost = X_test.iloc[:COST_N_INSTANCES]
    t0 = time.time()
    for _, row in X_cost.iterrows():
        explainer.explain_instance(row.values, model.predict_proba, num_features=N_LIME_FEATURES)
    lime_cost_seconds = time.time() - t0
    print(f"LIME: {lime_cost_seconds:.4f}s ({lime_cost_seconds / COST_N_INSTANCES * 1000:.2f} ms/instance)")

    print("\n" + "=" * 60)
    print("COMPARISON: SHAP vs. LIME")
    print("=" * 60)

    if os.path.exists(SHAP_COST_PATH):
        with open(SHAP_COST_PATH) as f:
            shap_cost = json.load(f)
        shap_seconds = shap_cost["shap_cost_seconds"]

        comparison = pd.DataFrame([
            {"method": "SHAP (TreeExplainer)", "seconds_per_100_instances": shap_seconds,
             "deterministic": True, "fidelity_guarantee": f"exact (max gap = {shap_cost['max_abs_fidelity_gap']:.2e})"},
            {"method": "LIME", "seconds_per_100_instances": lime_cost_seconds,
             "deterministic": False, "fidelity_guarantee": "none (local surrogate model)"},
        ])
        print(comparison.to_string(index=False))
        comparison.to_csv("outputs/results/xai_cost_comparison.csv", index=False)
        print(f"\nSHAP was {lime_cost_seconds / shap_seconds:.1f}x faster than LIME.")
    else:
        print(f"[warning] {SHAP_COST_PATH} not found, run 14_shap.py first.")

    print("\nLIME stage complete.")