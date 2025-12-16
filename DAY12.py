#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ===============================
# HANDS-ON TASK 1: KMEANS CLUSTERING
# Dataset: Synthetic / Simple Dataset
# ===============================

# STEP 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# STEP 1: Load & Inspect Dataset
# ===============================

# Create synthetic dataset
data = {
    "Feature1": [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    "Feature2": [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nShape:", df.shape)
print("\nDescription:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ===============================
# STEP 2: Feature Scaling
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ===============================
# STEP 3: Elbow Method
# ===============================

sse = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()

# ===============================
# STEP 4: Apply KMeans
# ===============================

# Choose K = 2 (based on elbow method)
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustered Data:")
print(df)

# ===============================
# STEP 5: Visualize Clusters
# ===============================

plt.figure()
plt.scatter(df["Feature1"], df["Feature2"], c=df["Cluster"])

# Plot centroids (convert back to original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("KMeans Clustering Visualization")
plt.show()


# In[4]:


# =====================================================
# KMEANS CLUSTERING – DATASET 2 (REAL-WORLD STYLE DATA)
# Customer Segmentation
# Features: Age, Annual_Income, Spending_Score
# =====================================================

# STEP 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# STEP 1: Create Real-World Style Customer Dataset
# =====================================================

np.random.seed(42)

data = {
    "Age": np.random.randint(18, 65, 100),
    "Annual_Income": np.random.randint(15000, 120000, 100),
    "Spending_Score": np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# =====================================================
# STEP 1: EDA (Exploratory Data Analysis)
# =====================================================

# Distribution of Annual Income
plt.figure()
plt.hist(df["Annual_Income"], bins=20)
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.title("Distribution of Annual Income")
plt.show()

# Distribution of Spending Score
plt.figure()
plt.hist(df["Spending_Score"], bins=20)
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.title("Distribution of Spending Score")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Outlier Detection using Boxplots
plt.figure()
sns.boxplot(data=df)
plt.title("Outlier Detection (Boxplot)")
plt.show()

# =====================================================
# STEP 2: Feature Scaling
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =====================================================
# STEP 3: Elbow Method
# =====================================================

sse = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method – Dataset 2")
plt.show()

# =====================================================
# STEP 4: Apply KMeans Clustering
# =====================================================

# Choose K = 4 (typical for customer segmentation)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustered Data Sample:")
print(df.head())

# =====================================================
# STEP 5: Interpret Clusters
# =====================================================

cluster_summary = df.groupby("Cluster").mean()
print("\nCluster Interpretation (Average Values):")
print(cluster_summary)

# =====================================================
# STEP 6: Visualize Clusters (Income vs Spending)
# =====================================================

plt.figure()
plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()


# In[6]:


# =====================================================
# PART 5: MINI PROJECT – MALL CUSTOMER SEGMENTATION
# =====================================================
# Goal:
# - Identify premium customers
# - Target promotions effectively
# - Improve sales strategy
# =====================================================

# STEP 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =====================================================
# STEP 1: Create Mall Customer Dataset (Synthetic)
# Typical columns:
# CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
# =====================================================

np.random.seed(42)

data = {
    "CustomerID": range(1, 201),
    "Gender": np.random.choice(["Male", "Female"], 200),
    "Age": np.random.randint(18, 70, 200),
    "Annual Income (k$)": np.random.randint(15, 150, 200),
    "Spending Score (1-100)": np.random.randint(1, 100, 200)
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# =====================================================
# STEP 1 — DATA UNDERSTANDING
# 1. Drop CustomerID
# 2. Encode Gender
# 3. Select meaningful features
# =====================================================

# Drop CustomerID (not useful for clustering)
df.drop(columns=["CustomerID"], inplace=True)

# Encode Gender (Male=1, Female=0)
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Select features for clustering
X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Features:")
print(X.head())

# =====================================================
# STEP 2 — SCALING
# Apply StandardScaler
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# STEP 3 — ELBOW METHOD
# Find optimal K
# =====================================================

sse = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method – Mall Customers")
plt.show()

# Justification:
# The elbow is usually observed around K = 5
# Hence, we choose K = 5

# =====================================================
# STEP 4 — KMEANS CLUSTERING
# Apply clustering and add cluster labels
# =====================================================

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustered Data Sample:")
print(df.head())

# =====================================================
# STEP 5 — PCA VISUALIZATION
# Reduce to 2D and plot clusters
# =====================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", pca.explained_variance_ratio_.sum())

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Mall Customer Segmentation (PCA Visualization)")
plt.show()

# =====================================================
# STEP 6 — CLUSTER INTERPRETATION (BUSINESS VIEW)
# =====================================================

cluster_summary = df.groupby("Cluster").mean()
print("\nCluster Summary (Average Values):")
print(cluster_summary)


# In[7]:


# =====================================================
# PART 4: APPLY PCA & COMPARE BEFORE / AFTER
# Goal:
# Understand how PCA affects:
# - Visualization
# - Cluster separation
# - Interpretability
# =====================================================

# STEP 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =====================================================
# STEP 1: Create Customer Dataset (Age, Income, Spending)
# =====================================================

np.random.seed(42)

data = {
    "Age": np.random.randint(18, 65, 150),
    "Annual_Income": np.random.randint(15, 120, 150),
    "Spending_Score": np.random.randint(1, 100, 150)
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# =====================================================
# STEP 2: Scaling
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =====================================================
# STEP 3: Apply KMeans BEFORE PCA
# =====================================================

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# =====================================================
# STEP 4: Visualization BEFORE PCA
# (Only 2 features can be shown at once)
# =====================================================

plt.figure()
plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Clusters BEFORE PCA")
plt.show()

# =====================================================
# STEP 5: Apply PCA (2 Components)
# =====================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", pca.explained_variance_ratio_.sum())

# =====================================================
# STEP 6: Visualization AFTER PCA
# =====================================================

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Clusters AFTER PCA")
plt.show()

# =====================================================
# STEP 7: Comparison Summary (Printed)
# =====================================================

print("\nCOMPARISON SUMMARY")
print("----------------------")
print("Before PCA:")
print("- Dimensions: 3 (Age, Income, Spending)")
print("- Visualization: Partial (2 features only)")
print("- Cluster separation: Moderate")

print("\nAfter PCA:")
print("- Dimensions: 2 (PC1, PC2)")
print("- Visualization: Complete & clearer")
print("- Cluster separation: Improved")


# In[8]:


# =====================================================
# ALGORITHMIC PRACTICE
# Algorithm 1: KMeans From Scratch (Logic Level)
# Algorithm 2: PCA Logic (Conceptual + Computation)
# Algorithm 3: Automated Elbow Finder
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# ALGORITHM 1 — KMEANS FROM SCRATCH
# =====================================================

# -------- Step 1: Create Sample Dataset --------
np.random.seed(42)
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

K = 2
max_iters = 100

# -------- Step 2: Distance Computation --------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------- Step 3: Initialize Centroids Randomly --------
centroids = X[np.random.choice(len(X), K, replace=False)]

# -------- Step 4: KMeans Loop --------
for iteration in range(max_iters):

    # Cluster Assignment
    clusters = [[] for _ in range(K)]
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Save old centroids for convergence check
    old_centroids = centroids.copy()

    # Centroid Update
    for i in range(K):
        centroids[i] = np.mean(clusters[i], axis=0)

    # Convergence Condition
    if np.all(old_centroids == centroids):
        print(f"KMeans converged at iteration {iteration}")
        break

print("\nFinal Centroids (KMeans from scratch):")
print(centroids)

# =====================================================
# ALGORITHM 2 — PCA LOGIC (CONCEPTUAL + COMPUTATION)
# =====================================================

# -------- Step 1: Standardization --------
# Why required?
# PCA is variance-based → features with larger scale dominate

X_meaned = X - np.mean(X, axis=0)

# -------- Step 2: Covariance Matrix --------
# Variance shows how much data spreads
cov_matrix = np.cov(X_meaned, rowvar=False)

# -------- Step 3: Eigen Decomposition --------
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# -------- Step 4: Sort Eigenvalues (Descending) --------
sorted_index = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_index]
eigen_vectors = eigen_vectors[:, sorted_index]

# -------- Step 5: Select First Principal Component --------
# First component explains maximum variance
principal_component = eigen_vectors[:, 0]

# -------- Step 6: Project Data --------
X_pca_1D = X_meaned.dot(principal_component)

print("\nEigenvalues (Variance per component):")
print(eigen_values)

print("\nFirst Principal Component Vector:")
print(principal_component)

# =====================================================
# ALGORITHM 3 — AUTOMATED ELBOW FINDER
# =====================================================

# -------- Step 1: Inertia Computation Function --------
def compute_inertia(X, centroids, labels):
    inertia = 0
    for i, point in enumerate(X):
        inertia += euclidean_distance(point, centroids[labels[i]]) ** 2
    return inertia

# -------- Step 2: Run KMeans for multiple K --------
def kmeans_inertia_for_k(X, K, iterations=50):
    centroids = X[np.random.choice(len(X), K, replace=False)]

    for _ in range(iterations):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, c) for c in centroids]
            labels.append(np.argmin(distances))

        labels = np.array(labels)

        for i in range(K):
            if len(X[labels == i]) > 0:
                centroids[i] = np.mean(X[labels == i], axis=0)

    return compute_inertia(X, centroids, labels)

# -------- Step 3: Calculate Inertia Values --------
Ks = range(1, 8)
inertias = [kmeans_inertia_for_k(X, k) for k in Ks]

# -------- Step 4: Detect Elbow Programmatically --------
# Method: Maximum second derivative (curvature)
second_derivative = np.diff(inertias, 2)
elbow_k = np.argmax(second_derivative) + 2

print("\nInertia values:", inertias)
print("Optimal K detected by automated elbow finder:", elbow_k)

# -------- Step 5: Plot Elbow --------
plt.figure()
plt.plot(Ks, inertias, marker='o')
plt.axvline(elbow_k, linestyle='--')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Automated Elbow Detection")
plt.show()


# In[9]:


# =====================================================
# AUTOMATED ELBOW FINDER + REAL-WORLD USE CASES
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# PART A — AUTOMATED ELBOW FINDER (LOGIC)
# =====================================================

def automated_elbow_finder(X, max_k=10):
    """
    Computes inertia for K=1..max_k
    Detects elbow using maximum curvature (2nd derivative)
    """
    inertias = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)

    # Second derivative to detect elbow
    curvature = np.diff(inertias, 2)
    optimal_k = np.argmax(curvature) + 2

    # Plot
    plt.figure()
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.axvline(optimal_k, linestyle='--')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Automated Elbow Detection")
    plt.show()

    return optimal_k, inertias

# =====================================================
# COMMON FUNCTION — KMEANS PIPELINE
# =====================================================

def run_kmeans_pipeline(data, feature_cols, title):
    df = pd.DataFrame(data, columns=feature_cols)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Elbow Finder
    k_opt, _ = automated_elbow_finder(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=k_opt, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print(f"\n{title}")
    print("Optimal K:", k_opt)
    print(df.groupby("Cluster").mean())

    return df

# =====================================================
# PART 7 — REAL-WORLD USE CASES
# =====================================================

# -----------------------------------------------------
# USE CASE 1: E-COMMERCE CUSTOMER SEGMENTATION
# -----------------------------------------------------
# Features: visits, cart value, purchases
# Goal: targeted ads

np.random.seed(1)
ecommerce_data = np.column_stack((
    np.random.randint(1, 50, 200),    # visits
    np.random.randint(100, 10000, 200), # cart value
    np.random.randint(1, 30, 200)     # purchases
))

ecommerce_df = run_kmeans_pipeline(
    ecommerce_data,
    ["Visits", "Cart_Value", "Purchases"],
    "E-Commerce Customer Segmentation"
)

# -----------------------------------------------------
# USE CASE 2: BANK CUSTOMER GROUPING
# -----------------------------------------------------
# Features: balance, transactions, credit usage
# Goal: risk analysis

bank_data = np.column_stack((
    np.random.randint(1000, 500000, 200),  # balance
    np.random.randint(1, 200, 200),        # transactions
    np.random.randint(0, 100, 200)         # credit usage %
))

bank_df = run_kmeans_pipeline(
    bank_data,
    ["Balance", "Transactions", "Credit_Usage"],
    "Bank Customer Risk Grouping"
)

# -----------------------------------------------------
# USE CASE 3: STUDENT BEHAVIOR CLUSTERING
# -----------------------------------------------------
# Features: attendance, marks, participation
# Goal: personalized mentoring

student_data = np.column_stack((
    np.random.randint(40, 100, 150),   # attendance %
    np.random.randint(35, 100, 150),   # marks
    np.random.randint(1, 10, 150)      # participation
))

student_df = run_kmeans_pipeline(
    student_data,
    ["Attendance", "Marks", "Participation"],
    "Student Behavior Clustering"
)

# -----------------------------------------------------
# USE CASE 4: SOCIAL MEDIA USER CLUSTERS
# -----------------------------------------------------
# Features: likes, comments, shares
# Goal: influencer detection

social_data = np.column_stack((
    np.random.randint(10, 10000, 200),   # likes
    np.random.randint(1, 2000, 200),     # comments
    np.random.randint(1, 3000, 200)      # shares
))

social_df = run_kmeans_pipeline(
    social_data,
    ["Likes", "Comments", "Shares"],
    "Social Media User Clustering"
)


# In[ ]:




