"""
Loading and inspecting the Credit Card Fraud Dataset
-----------------------------------------------------
Dataset:
- European cardholders, September 2013
- 284,807 transactions
- 492 fraudulent transactions (highly imbalanced dataset)
- 31 columns

Features:
- V1–V28: PCA-transformed features (original features hidden for privacy)
- Time: Seconds elapsed since the first transaction
- Amount: Transaction amount
- Class: Target variable
    0 = Genuine transaction
    1 = Fraudulent transaction
"""

# Import standard Python modules
import os
from pathlib import Path

# --------------------------------------------------
# Project Directory Setup
# --------------------------------------------------

# Get the project root directory.
# Example:
# src/01_load_data.py
# Project Root/
# ├── data/
# ├── outputs/
# └── src/
#
# This allows the script to work no matter where it is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change current working directory to project root
# so relative paths such as data/ and outputs/ work correctly.
os.chdir(_PROJECT_ROOT)

# Create output folders automatically if they do not already exist.
# This prevents errors when saving plots, models, or results later.
for _sub in ["plots", "models", "results", "interim"]:
    os.makedirs(f"outputs/{_sub}", exist_ok=True)

# --------------------------------------------------
# Data Loading
# --------------------------------------------------

# Pandas is the main library used for data analysis.
import pandas as pd

# Location of the dataset inside the project folder.
DATA_PATH = "data/creditcard.csv"


def load_data(path=DATA_PATH, drop_duplicates=True):
    """
    Load the credit card fraud dataset.

    Parameters
    ----------
    path : str
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    # Read CSV file into a Pandas DataFrame.
    df = pd.read_csv(path)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"[load_data] Removed {removed} duplicate rows "
                  f"({removed / before:.3%} of the dataset) before any further processing.")

    return df


# --------------------------------------------------
# Main Program Execution
# --------------------------------------------------

# This block only runs when the file is executed directly.
# It will not run if this file is imported into another script.
if __name__ == "__main__":

    # Load dataset into memory.
    df = load_data()

    # --------------------------------------------------
    # Basic Dataset Information
    # --------------------------------------------------

    # Number of rows and columns.
    # Example output: (284807, 31)
    print("Shape:", df.shape)

    # Display data types of all columns.
    # Helps identify numeric, categorical, or incorrect data types.
    print("\nColumn dtypes:\n", df.dtypes)

    # Count missing values in the entire dataset.
    # A clean dataset should return 0.
    print(
        "\nMissing values per column:\n",
        df.isnull().sum().sum(),
        "total missing values"
    )

    # --------------------------------------------------
    # Target Variable Analysis
    # --------------------------------------------------

    print("\nClass balance:")

    # Display actual number of fraud and non-fraud records.
    print(df["Class"].value_counts())

    # Display percentage distribution of classes.
    # Useful for understanding class imbalance.
    print(df["Class"].value_counts(normalize=True) * 100)

    # --------------------------------------------------
    # Transaction Amount Analysis
    # --------------------------------------------------

    # Summary statistics:
    # count, mean, std, min, max, quartiles
    print("\nAmount stats:\n", df["Amount"].describe())

    # ------------------------------------------
