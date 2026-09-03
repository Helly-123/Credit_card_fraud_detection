# Credit Card Fraud Detection Project

Implementation of machine learning, deep learning, and explainable AI (XAI) techniques for credit card fraud detection using the Kaggle Credit Card Fraud Detection Dataset.

---

# Project Structure

```text
CCF_Fraud_Detection_Project/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── 01_load_data.py
│   ├── 02_eda.py
│   ├── 03_split_and_scale.py
│   ├── 04_resample_smote.py
│   ├── 05_train_models.py
│   ├── 06_optimize_model.py
│   ├── 07_linreg_svm_nb.py
│   ├── 08_kmeans.py
│   ├── 09_dimensionality_reduction.py
│   ├── 10_gradient_boosting.py
│   ├── 11_cnn.py
│   ├── 12_mlp.py
│   ├── 13_lstm.py
│   ├── 14_shap.py
│   ├── 15_lime.py
│   └── eval_utils.py
│
├── outputs/
│   ├── interim/
│   ├── models/
│   ├── plots/
│   └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Script Overview

## Data Preparation

### 01_load_data.py
- Loads the credit card fraud dataset
- Performs basic validation and sanity checks
- Displays dataset information and summary statistics

### 02_eda.py
- Exploratory Data Analysis (EDA)
- Class distribution analysis
- Transaction amount analysis
- Transaction time analysis
- Correlation analysis
- Generates EDA visualisations

### 03_split_and_scale.py
- Performs stratified train-test split
- Preserves original fraud class distribution
- Scales Time and Amount features
- Saves processed datasets

### 04_resample_smote.py
- Applies SMOTE to training data only
- Balances the minority fraud class
- Prevents data leakage into the test set

---

## Machine Learning Models

### 05_train_models.py
Trains and evaluates:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 06_optimize_model.py
- Hyperparameter tuning
- Cross-validation
- Best model optimisation

### 07_linreg_svm_nb.py
Trains and evaluates:
- Linear Regression (threshold-based classification)
- Support Vector Machine (SVM)
- Naive Bayes

### 08_kmeans.py
- K-Means clustering
- Unsupervised fraud detection analysis

### 09_dimensionality_reduction.py
- Principal Component Analysis (PCA)
- Performance comparison before and after dimensionality reduction

### 10_gradient_boosting.py
- Gradient Boosting Classifier
- Hyperparameter optimisation
- Performance evaluation

---

## Deep Learning Models

### 11_cnn.py
- One-Dimensional Convolutional Neural Network (1D CNN)
- TensorFlow/Keras implementation
- Fraud classification using convolutional layers

### 12_mlp.py
- Multi-Layer Perceptron (MLP)
- Fully connected feed-forward neural network
- Fraud classification using dense hidden layers

### 13_lstm.py
- Long Short-Term Memory Network (LSTM)
- Bidirectional recurrent architecture
- Deep learning fraud detection model

---

## Explainable AI (XAI)

### 14_shap.py
- SHAP (SHapley Additive exPlanations)
- Global feature importance analysis
- Local prediction explanations
- Model interpretability visualisations

### 15_lime.py
- LIME (Local Interpretable Model-Agnostic Explanations)
- Instance-level explanations
- Local prediction interpretation

---

## Utility Module

### eval_utils.py

Shared helper functions for:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Reports
- Model evaluation and comparison

---

# Installation

Navigate to the project directory:

```bash
cd CCF_Fraud_Detection_Project
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Run the scripts in numerical order.

## Data Preparation

```bash
python src/01_load_data.py

python src/02_eda.py

python src/03_split_and_scale.py

python src/04_resample_smote.py
```

## Machine Learning Models

```bash
python src/05_train_models.py

python src/06_optimize_model.py

python src/07_linreg_svm_nb.py

python src/08_kmeans.py

python src/09_dimensionality_reduction.py

python src/10_gradient_boosting.py
```

## Deep Learning Models

```bash
python src/11_cnn.py

python src/12_mlp.py

python src/13_lstm.py
```

## Explainable AI

```bash
python src/14_shap.py

python src/15_lime.py
```

---

# Outputs

All generated files are stored under:

```text
outputs/
│
├── interim/
│   ├── train-test splits
│   ├── scaled datasets
│   └── SMOTE-balanced datasets
│
├── models/
│   ├── Trained machine learning models
│   ├── Deep learning models (.keras)
│   └── Tuned model artifacts
│
├── plots/
│   ├── EDA visualisations
│   ├── Performance charts
│   ├── SHAP plots
│   └── LIME explanations
│
└── results/
    ├── Evaluation metrics
    ├── Comparison tables
    └── Summary reports
```

---

# Recommended Execution Order

```text
01 → 02 → 03 → 04
         ↓
05 → 06 → 07 → 08 → 09 → 10
         ↓
11 → 12 → 13
         ↓
14 → 15
```

This workflow ensures data preparation is completed before machine learning training, deep learning experiments, and explainability analysis.