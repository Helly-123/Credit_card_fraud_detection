"""
ALGORITHM 6: CONVOLUTIONAL NEURAL NETWORK (1D CNN)
------------------------------------------------------
This experiment evaluates a 1D Convolutional Neural Network (CNN) for
credit card fraud detection. Each transaction is reshaped from a
30-feature vector into a sequence of shape (30,1), allowing Conv1D
layers to learn local feature interactions automatically.

Pipeline:
    StandardScaler
        ↓
    SMOTE
        ↓
    Conv1D + BatchNorm + Dropout
        ↓
    Dense Layers
        ↓
    Sigmoid Output

Two CNN configurations are evaluated using EarlyStopping and the best
architecture is selected based on validation PR-AUC. Final evaluation
is performed on the full untouched test set.

Key Metric:
    PR-AUC (most appropriate for imbalanced fraud detection datasets)

Outputs:
    outputs/results/results_cnn.csv
    outputs/models/model_cnn.keras
"""

import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from eval_utils import load_all, evaluate

tf.random.set_seed(42)
np.random.seed(42)


def build_cnn(input_dim, n_conv_blocks=2, dropout=0.3, lr=1e-3):
    inputs = keras.Input(shape=(input_dim, 1))
    x = inputs
    filters = 32
    for i in range(n_conv_blocks):
        x = layers.Conv1D(filters, kernel_size=2, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout if i == n_conv_blocks - 1 else dropout * 0.6)(x)
        filters *= 2
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR"), "accuracy"],
    )
    return model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_train_s, y_train)

    # Subsample the balanced set for CPU-time reasons
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_bal), size=min(60000, len(X_bal)), replace=False)
    X_bal, y_bal = X_bal[idx], y_bal.values[idx] if hasattr(y_bal, "values") else y_bal[idx]

    X_bal_cnn = X_bal.reshape(-1, X_bal.shape[1], 1)
    X_test_cnn = X_test_s.reshape(-1, X_test_s.shape[1], 1)

    # Small architecture search 
    configs = [
        {"n_conv_blocks": 2, "dropout": 0.3, "lr": 1e-3},
        {"n_conv_blocks": 3, "dropout": 0.3, "lr": 5e-4},
    ]

    results = []
    best_model, best_pr = None, -1
    for i, cfg in enumerate(configs):
        print(f"\n--- CNN config {i+1}: {cfg} ---")
        model = build_cnn(X_bal.shape[1], **cfg)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True
        )
        history = model.fit(
            X_bal_cnn, y_bal,
            validation_split=0.15,
            epochs=15,
            batch_size=512,
            callbacks=[early_stop],
            verbose=2,
        )
        val_pr = max(history.history["val_pr_auc"])
        print(f"Config {i+1} best val PR-AUC: {val_pr:.4f}")
        if val_pr > best_pr:
            best_pr, best_model, best_cfg = val_pr, model, cfg

    print(f"\nBest CNN config: {best_cfg} (val PR-AUC={best_pr:.4f})")

    y_score = best_model.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_score >= 0.5).astype(int)
    result = evaluate("CNN (1D Conv, tuned architecture)", y_test, y_pred, y_score, note=str(best_cfg))

    pd.DataFrame([result]).to_csv("outputs/results/results_cnn.csv", index=False)
    best_model.save("outputs/models/model_cnn.keras")
    print("\nSaved outputs/results/results_cnn.csv and outputs/models/model_cnn.keras")
