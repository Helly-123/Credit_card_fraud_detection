"""
ALGORITHM 12: LONG SHORT-TERM MEMORY NETWORK (LSTM, TensorFlow/Keras)
--------------------------------------------------------------------
A Long Short-Term Memory (LSTM) network is a specialised recurrent neural
network (RNN) designed to capture complex sequential dependencies through
its memory-cell architecture and gated information flow mechanisms.
Although the credit card fraud dataset is fundamentally tabular rather
than temporal, each transaction's 30 features are reshaped into a
pseudo-sequence of shape (30, 1), enabling the application of recurrent
deep learning architectures as commonly reported in fraud detection
literature.

Unlike the MLP, which processes all features simultaneously as a flat
vector, the LSTM sequentially examines feature values and learns
higher-order non-linear relationships through its input, forget, and
output gates. This approach allows recurrent layers to model interactions
between adjacent feature representations, even when no true temporal
ordering exists.

Architecture:

  • Input sequence: (30, 1)

  • Stacked Bidirectional LSTM layers, allowing information to be
    processed in both forward and backward directions.

  • Batch Normalization and Dropout layers for regularisation and
    improved generalisation.

  • Fully connected Dense classification head.

  • Sigmoid output layer producing the probability of fraud.

Methodology follows the same leakage-safe workflow used throughout this
project:

  • SMOTE is applied exclusively to training data after the train-test
    split, preventing information leakage from the test set.

  • Model development, tuning, and training are performed using only
    training data.

  • Final evaluation is conducted on the original, untouched hold-out
    test set, preserving the real-world class imbalance and providing an
    unbiased estimate of model performance.

Computational Considerations:

  • LSTM networks are the most computationally expensive models in this
    project because recurrent operations must be executed sequentially,
    limiting parallelisation compared with feed-forward architectures.

  • Training is therefore performed on a smaller SMOTE-balanced training
    subset to maintain practical execution times within the available
    CPU-only environment.

  • EarlyStopping is used to reduce unnecessary training epochs and
    mitigate overfitting.

  • CuDNN-specific configurations and recurrent dropout are avoided to
    maintain efficient execution and compatibility across different
    hardware environments.

Despite being originally designed for sequential data, LSTM networks are
frequently evaluated in fraud detection research alongside CNN and MLP
architectures to investigate whether deep recurrent representations can
improve the identification of complex fraudulent transaction patterns.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from eval_utils import load_all, evaluate

tf.random.set_seed(42)
np.random.seed(42)


def build_lstm(input_dim, n_lstm_layers=2, units=64, dropout=0.3, lr=1e-3):
    """
    Stacked Bidirectional LSTM classifier.

    input_dim     -- number of features (treated as sequence length, 1 value/step)
    n_lstm_layers -- how many (Bi)LSTM layers to stack
    units         -- hidden units per LSTM direction (halved each layer after the first)
    dropout       -- dropout rate applied after each LSTM layer and the dense head
    lr            -- Adam learning rate
    """
    inputs = keras.Input(shape=(input_dim, 1))
    x = inputs
    layer_units = units
    for i in range(n_lstm_layers):
        return_seq = i < n_lstm_layers - 1  # only the last LSTM layer collapses the sequence
        x = layers.Bidirectional(
            layers.LSTM(layer_units, return_sequences=return_seq)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout if i == n_lstm_layers - 1 else dropout * 0.6)(x)
        layer_units = max(16, layer_units // 2)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
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

   # Create a dedicated threshold-tuning subset from the original training
# data before any resampling is performed. Stratified sampling preserves
# the natural class imbalance (~0.17% fraud prevalence), providing a
# realistic environment for threshold optimisation.
#
# This subset is neither used for model fitting nor final evaluation. It
# serves only to identify an appropriate decision threshold, ensuring
# that threshold selection remains independent of the hold-out test set
# and free from data leakage.

    X_fit_raw, X_thresh_raw, y_fit_raw, y_thresh_raw = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit_raw)
    X_thresh_s = scaler.transform(X_thresh_raw)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_fit_s, y_fit_raw)

    # Subsample the balanced set -- LSTMs are the slowest architecture here
    # per-epoch, so we cap this well below the CNN script's 60k rows.
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_bal), size=min(40000, len(X_bal)), replace=False)
    X_bal, y_bal = X_bal[idx], y_bal.values[idx] if hasattr(y_bal, "values") else y_bal[idx]

    X_bal_lstm = X_bal.reshape(-1, X_bal.shape[1], 1)
    X_thresh_lstm = X_thresh_s.reshape(-1, X_thresh_s.shape[1], 1)
    X_test_lstm = X_test_s.reshape(-1, X_test_s.shape[1], 1)

    # Small architecture search 
    configs = [
        {"n_lstm_layers": 2, "units": 64, "dropout": 0.3, "lr": 1e-3},
        {"n_lstm_layers": 2, "units": 96, "dropout": 0.3, "lr": 5e-4},
    ]

    best_model, best_pr, best_cfg = None, -1, None
    for i, cfg in enumerate(configs):
        print(f"\n--- LSTM config {i + 1}: {cfg} ---")
        model = build_lstm(X_bal.shape[1], **cfg)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True
        )
        history = model.fit(
            X_bal_lstm, y_bal,
            validation_split=0.15,
            epochs=20,
            batch_size=512,
            callbacks=[early_stop],
            verbose=2,
        )
        val_pr = max(history.history["val_pr_auc"])
        print(f"Config {i + 1} best val PR-AUC: {val_pr:.4f}")
        if val_pr > best_pr:
            best_pr, best_model, best_cfg = val_pr, model, cfg

    print(f"\nBest LSTM config: {best_cfg} (val PR-AUC={best_pr:.4f})")

  # --- Decision-threshold optimisation on a real-imbalance tuning set ---
# Although training is performed on SMOTE-balanced data, the optimal
# classification threshold is not assumed to be 0.50 because the original
# fraud prevalence is extremely low (~0.17%). A fixed 0.50 cutoff often
# results in poor precision and excessive false-positive predictions when
# applied to real-world data.
#
# Predicted probabilities are therefore evaluated across a range of
# thresholds on an untouched tuning subset that preserves the natural
# class distribution. The threshold yielding the highest Fraud-F1 score
# is selected and subsequently applied during final evaluation. This
# provides a data-driven operating point while preserving the integrity
# of the independent test set.

    thresh_scores = best_model.predict(X_thresh_lstm, verbose=0).ravel()
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_thresh_raw, (thresh_scores >= t).astype(int), zero_division=0)
           for t in candidate_thresholds]
    best_threshold = candidate_thresholds[int(np.argmax(f1s))]
    print(f"Tuned decision threshold: {best_threshold:.2f} "
          f"(F1={max(f1s):.4f} on the held-out real-imbalance tuning slice, "
          f"vs default 0.50)")

    y_score = best_model.predict(X_test_lstm, verbose=0).ravel()
    y_pred = (y_score >= best_threshold).astype(int)
    result = evaluate("LSTM (Bidirectional, tuned architecture + tuned threshold)",
                       y_test, y_pred, y_score,
                       note=f"{best_cfg}, threshold={best_threshold:.2f}")

    pd.DataFrame([result]).to_csv("outputs/results/results_lstm.csv", index=False)
    best_model.save("outputs/models/model_lstm.keras")
    print("\nSaved outputs/results/results_lstm.csv and outputs/models/model_lstm.keras")
