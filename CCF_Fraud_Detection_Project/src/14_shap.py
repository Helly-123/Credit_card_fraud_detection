"""
STEP 14: SHAP (SHapley Additive exPlanations)
-----------------------------------------------------------------------------
Uses outputs/models/model_xgboost.joblib (untuned XGBoost from
05_train_models.py, PR-AUC 0.8206). 

Produces:
  - Global feature importance (beeswarm + bar plots)
  - Local explanations for a true positive, false positive, false negative
  - A fidelity check (SHAP values should exactly reconstruct model output)
  - A cost timing, saved for ime to compare against
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
import matplotlib.pyplot as plt
import shap

from eval_utils import load_all

RANDOM_STATE = 42
MODEL_PATH = "outputs/models/model_xgboost.joblib"
SAMPLE_SIZE = 5000
TOP_N_FEATURES = 15
COST_N_INSTANCES = 100


def build_shap_sample(X_test, y_test, sample_size, random_state):
    # keep every fraud row so TP/FP/FN examples are always findable
    rng = np.random.RandomState(random_state)
    fraud_idx = y_test[y_test == 1].index
    legit_idx = y_test[y_test == 0].index

    n_legit_needed = max(sample_size - len(fraud_idx), 0)
    legit_sample_idx = rng.choice(legit_idx, size=min(n_legit_needed, len(legit_idx)), replace=False)

    sample_idx = np.concatenate([fraud_idx.to_numpy(), legit_sample_idx])
    rng.shuffle(sample_idx)
    return X_test.loc[sample_idx], y_test.loc[sample_idx]


def find_example_indices(X_sample, y_sample, y_pred_sample):
    y_pred_sample = pd.Series(y_pred_sample, index=y_sample.index)

    tp = y_sample[(y_sample == 1) & (y_pred_sample == 1)].index
    fp = y_sample[(y_sample == 0) & (y_pred_sample == 1)].index
    fn = y_sample[(y_sample == 1) & (y_pred_sample == 0)].index

    examples = {
        "true_positive": tp[0] if len(tp) else None,
        "false_positive": fp[0] if len(fp) else None,
        "false_negative": fn[0] if len(fn) else None,
    }
    for name, idx in examples.items():
        if idx is None:
            print(f"[warning] No {name} example found in this sample.")
    return examples


if __name__ == "__main__":
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    X_train, y_train, X_test, y_test = load_all()

    X_sample, y_sample = build_shap_sample(X_test, y_test, SAMPLE_SIZE, RANDOM_STATE)
    print(f"SHAP sample: {X_sample.shape[0]} rows "
          f"({int(y_sample.sum())} fraud, {int((y_sample == 0).sum())} legit)")

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_sample)

    # ---- Global explanation ----
    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE (SHAP)")
    print("=" * 60)

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)

    print(f"\nTop {TOP_N_FEATURES} features by mean |SHAP value|:")
    print(mean_abs_shap.head(TOP_N_FEATURES).to_string())
    mean_abs_shap.rename("mean_abs_shap_value").to_csv("outputs/results/shap_top_features.csv")

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.tight_layout()
    plt.savefig("outputs/plots/shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.tight_layout()
    plt.savefig("outputs/plots/shap_bar_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved shap_beeswarm.png, shap_bar_importance.png, shap_top_features.csv")

    # ---- Local explanations ----
    print("\n" + "=" * 60)
    print("LOCAL EXPLANATIONS")
    print("=" * 60)

    y_pred_sample = model.predict(X_sample)
    examples = find_example_indices(X_sample, y_sample, y_pred_sample)

    # saved so 15_lime.py explains the same rows
    with open("outputs/interim/xai_example_indices.json", "w") as f:
        json.dump({
            "sample_indices": [int(i) for i in X_sample.index],
            "examples": {k: (int(v) if v is not None else None) for k, v in examples.items()},
        }, f, indent=2)
    print("Saved outputs/interim/xai_example_indices.json")

    position_lookup = {idx: pos for pos, idx in enumerate(X_sample.index)}

    for case_name, idx in examples.items():
        if idx is None:
            continue
        pos = position_lookup[idx]
        print(f"\n--- {case_name} (test row index {idx}) ---")
        print(f"True label: {int(y_sample.loc[idx])} | "
              f"Predicted: {int(y_pred_sample[pos])} | "
              f"Fraud probability: {model.predict_proba(X_sample.loc[[idx]])[0, 1]:.4f}")

        plt.figure()
        shap.plots.waterfall(shap_values[pos], max_display=TOP_N_FEATURES, show=False)
        plt.tight_layout()
        out_path = f"outputs/plots/shap_waterfall_{case_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    # ---- Fidelity check (RQ3) ----
    print("\n" + "=" * 60)
    print("FIDELITY CHECK")
    print("=" * 60)

    raw_margin = model.predict(X_sample, output_margin=True)
    shap_reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
    fidelity_gap = np.abs(raw_margin - shap_reconstructed)

    print(f"Mean absolute gap: {fidelity_gap.mean():.10f}")
    print(f"Max absolute gap:  {fidelity_gap.max():.10f}")

    # ---- Cost timing  ----
    print("\n" + "=" * 60)
    print(f"COST: explaining {COST_N_INSTANCES} instances with SHAP")
    print("=" * 60)

    X_cost = X_sample.iloc[:COST_N_INSTANCES]
    t0 = time.time()
    _ = explainer(X_cost)
    shap_cost_seconds = time.time() - t0
    print(f"SHAP: {shap_cost_seconds:.4f}s ({shap_cost_seconds / COST_N_INSTANCES * 1000:.2f} ms/instance)")

    with open("outputs/results/shap_cost_and_fidelity.json", "w") as f:
        json.dump({
            "model": "XGBoost (untuned, SMOTE-trained, from 05_train_models.py)",
            "n_instances_timed": COST_N_INSTANCES,
            "shap_cost_seconds": shap_cost_seconds,
            "mean_abs_fidelity_gap": float(fidelity_gap.mean()),
            "max_abs_fidelity_gap": float(fidelity_gap.max()),
        }, f, indent=2)
    print("Saved outputs/results/shap_cost_and_fidelity.json")

    print("\nSHAP stage complete.")