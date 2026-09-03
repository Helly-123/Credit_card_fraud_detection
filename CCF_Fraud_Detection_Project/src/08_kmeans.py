"""
ALGORITHM: K-MEANS CLUSTERING
------------------------------------------------------
K-Means is an unsupervised learning algorithm, meaning it does not use
class labels (fraud/non-fraud) during model training. Instead, it groups
transactions into clusters based solely on feature similarity and distance
in the feature space.

To apply K-Means to fraud detection, the workflow is:

  1. Scale the training data and fit K-Means using only the feature
     variables, without access to the fraud labels.

  2. After clustering is complete, examine the fraud distribution within
     each cluster using the training labels. The cluster with the highest
     concentration of fraudulent transactions is designated as the
     "fraud-leaning" cluster.

  3. Assign unseen transactions to their nearest cluster centroid and
     classify transactions belonging to the fraud-leaning cluster as
     potential fraud cases.

This approach mirrors a realistic semi-supervised scenario where clusters
are discovered automatically and only a small amount of labeled data is
used afterwards to interpret cluster meaning.

Optimization and Analysis:
  - Select the optimal number of clusters (k) using methods such as the
    Elbow Method and Silhouette Score.
  - Compare clustering performance on the original feature space versus
    PCA-reduced feature space. Dimensionality reduction can improve the
    effectiveness of distance-based clustering by reducing noise,
    redundancy, and the effects of high dimensionality.

Limitations:
  - K-Means assumes spherical clusters of similar density and may struggle
    with highly imbalanced datasets such as credit card fraud detection.
  - Since fraud cases typically represent a very small minority of
    transactions, clusters may not naturally separate fraudulent and
    legitimate behavior as clearly as supervised learning methods.
  - Results are sensitive to the chosen value of k and the initial
    centroid placement.

Despite these limitations, K-Means provides a useful unsupervised
baseline and enables investigation of whether fraudulent transactions
form naturally distinguishable groups within the dataset.
"""
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)  # makes data/ and outputs/ paths work from any launch location
for _sub in ['plots', 'models', 'results', 'interim']:
    os.makedirs(f'outputs/{_sub}', exist_ok=True)


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from eval_utils import load_all, evaluate

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all()
    results = []

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Elbow-style check across k (fit on a subsample for speed) ----
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train_s), size=30000, replace=False)
    X_sub = X_train_s[sub_idx]

    print("Inertia by k (elbow method, lower isn't always better -- look for the bend):")
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X_sub)
        print(f"  k={k}: inertia={km.inertia_:.1f}")

    # k=2 is the natural choice for binary fraud/legit framing
    print("\nUsing k=2 (matches the fraud vs. legit binary problem)")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(X_train_s)

    train_clusters = kmeans.predict(X_train_s)
    fraud_rate_per_cluster = pd.Series(y_train.values).groupby(train_clusters).mean()
    print("Fraud rate per cluster (train):\n", fraud_rate_per_cluster)
    fraud_cluster = fraud_rate_per_cluster.idxmax()
    print(f"Cluster {fraud_cluster} is treated as the 'fraud' cluster "
          f"(fraud rate {fraud_rate_per_cluster[fraud_cluster]*100:.3f}% vs "
          f"{fraud_rate_per_cluster[1-fraud_cluster]*100:.3f}%)")

    test_clusters = kmeans.predict(X_test_s)
    y_pred = (test_clusters == fraud_cluster).astype(int)
    # distance-to-fraud-centroid as a pseudo score (closer = more fraud-like)
    dist_to_centroids = kmeans.transform(X_test_s)
    y_score = -dist_to_centroids[:, fraud_cluster]  # negative distance -> higher = closer

    results.append(evaluate("K-Means (raw features, k=2)", y_test, y_pred, y_score,
                             note="unsupervised clustering, cluster->label mapping via train fraud rate"))

    # ---- Same idea but cluster on PCA-reduced space (often cleaner clusters) ----
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    kmeans_pca = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans_pca.fit(X_train_pca)
    train_clusters_pca = kmeans_pca.predict(X_train_pca)
    fraud_rate_pca = pd.Series(y_train.values).groupby(train_clusters_pca).mean()
    fraud_cluster_pca = fraud_rate_pca.idxmax()
    print(f"\n[PCA-10 clustering] fraud cluster {fraud_cluster_pca}, "
          f"rate {fraud_rate_pca[fraud_cluster_pca]*100:.3f}%")

    test_clusters_pca = kmeans_pca.predict(X_test_pca)
    y_pred_pca = (test_clusters_pca == fraud_cluster_pca).astype(int)
    dist_pca = kmeans_pca.transform(X_test_pca)
    y_score_pca = -dist_pca[:, fraud_cluster_pca]

    results.append(evaluate("K-Means (PCA-10 features, k=2)", y_test, y_pred_pca, y_score_pca,
                             note="clustering on 10 PCA components instead of raw 30 features"))

    pd.DataFrame(results).to_csv("outputs/results/results_kmeans.csv", index=False)
    print("\nSaved outputs/results/results_kmeans.csv")
    print("\nNOTE: K-Means is not expected to beat supervised models here -- it has "
          "no access to labels during fitting. It's included because the paper lists "
          "it, and the low recall/precision below is the expected, honest outcome.")
