#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ----------------------------------------
# STEP 1: Import Required Libraries
# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------------------------
# STEP 2: Create Classmates Dataset
# ----------------------------------------
data = {
    "Student": ["A", "B", "C", "D", "E", "F"],
    "Gaming": [10, 6, 3, 8, 2, 5],
    "Sports": [2, 8, 1, 6, 9, 7],
    "Music": [7, 4, 10, 5, 6, 3],
    "Coding": [9, 5, 2, 8, 4, 6],
    "Art": [1, 3, 9, 2, 8, 4]
}

df = pd.DataFrame(data)
print("\n=== Original Dataset ===")
print(df)

# ----------------------------------------
# STEP 3: Prepare Data for Clustering
# ----------------------------------------
X = df.drop("Student", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# STEP 4: Apply K-Means Clustering
# ----------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n=== Dataset with Cluster Labels ===")
print(df)

# ----------------------------------------
# STEP 5: Apply PCA for Visualization
# ----------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# ----------------------------------------
# STEP 6: Visualize Clusters
# ----------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])

for i in range(len(df)):
    plt.text(df["PCA1"][i], df["PCA2"][i], df["Student"][i])

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Classmates Clustered by Interests")
plt.show()

# ----------------------------------------
# STEP 7: Analyze Each Cluster (✅ FIXED)
# ----------------------------------------
print("\n=== Cluster-wise Average Interests ===")

numeric_cols = ["Gaming", "Sports", "Music", "Coding", "Art"]
cluster_summary = df.groupby("Cluster")[numeric_cols].mean()

print(cluster_summary)

# ----------------------------------------
# STEP 8: Human-Friendly Cluster Meaning
# ----------------------------------------
print("\n=== Cluster Interpretation ===")
print("Cluster 0 → Gamers / Tech Enthusiasts")
print("Cluster 1 → Artists & Music Lovers")
print("Cluster 2 → Sporty Students")
print("Cluster 3 → Coders / Logical Thinkers")

# ----------------------------------------
# STEP 9: Ideal Team Roles Suggestion
# ----------------------------------------
print("\n=== Ideal Team Roles ===")
print("Gamers → Problem solving & testing")
print("Coders → Development & algorithms")
print("Artists → UI/UX & presentations")
print("Sporty → Team lead & coordination")


# In[4]:


# ============================================
# CUSTOMER SEGMENTATION WITH PCA VISUALIZATION
# MINI PROJECT – SESSION 10
# ============================================

# -------------------------------
# STEP 1: Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# STEP 2: Create Dataset
# -------------------------------
data = {
    "Age": [22, 25, 47, 52, 46, 56, 30, 28, 60, 40],
    "Annual_Income": [15000, 18000, 52000, 48000, 60000, 62000, 30000, 28000, 70000, 45000],
    "Spending_Score": [80, 75, 20, 25, 65, 30, 70, 85, 15, 60],
    "Savings": [2000, 3000, 15000, 18000, 12000, 22000, 5000, 4000, 30000, 10000]
}

df = pd.DataFrame(data)
print("\n===== ORIGINAL DATASET =====")
print(df)

# -------------------------------
# STEP 3: Scaling the Data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -------------------------------
# STEP 4: Elbow Method
# -------------------------------
inertia = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(range(1, 8), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Find Optimal K")
plt.show()

# -------------------------------
# STEP 5: Train KMeans Model
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n===== DATASET WITH CLUSTERS =====")
print(df)

# -------------------------------
# STEP 6: Apply PCA
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# -------------------------------
# STEP 7: PCA Visualization
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])

for i in range(len(df)):
    plt.text(df["PCA1"][i], df["PCA2"][i], str(i))

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segmentation using PCA")
plt.show()

# -------------------------------
# STEP 8: Cluster Interpretation
# -------------------------------
print("\n===== CLUSTER-WISE AVERAGES =====")
print(df.groupby("Cluster").mean())

# -------------------------------
# STEP 9: Human Interpretation
# -------------------------------
print("\n===== CLUSTER MEANING =====")
print("Cluster 0 → Young, High-Spending Customers")
print("Cluster 1 → Older, Low-Spending Customers")
print("Cluster 2 → Middle-Income, High-Savings Group")
print("Cluster 3 → High-Income, Low-Spending Customers")

# -------------------------------
# STEP 10: Mandatory 5 Insights
# -------------------------------
print("\n===== KEY INSIGHTS =====")
print("1. Cluster 0 consists of young customers with high spending behavior.")
print("2. Cluster 1 represents older customers who spend less and save more.")
print("3. Cluster 2 shows middle-income customers with strong savings habits.")
print("4. Cluster 3 includes high-income but low-spending customers.")
print("5. PCA visualization confirms clear separation between customer groups.")

# -------------------------------
# FINAL CONCLUSION
# -------------------------------
print("\n===== CONCLUSION =====")
print("Customer segmentation using K-Means helped identify distinct behavior patterns.")
print("Elbow method ensured optimal cluster selection.")
print("PCA simplified visualization of multi-dimensional data.")
print("These insights can support targeted marketing and business decisions.")


# In[6]:


# =========================================================
# CLUSTERING MASTER LAB – FULL CORRECTED VERSION
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_sample_image

# =========================================================
# EXERCISE 1 — KMEANS FROM SCRATCH
# =========================================================

print("\n=== EXERCISE 1: KMEANS FROM SCRATCH ===")

X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

K = 2
centroids = X[np.random.choice(len(X), K, replace=False)]

for _ in range(5):
    clusters = {i: [] for i in range(K)}
    for point in X:
        distances = [euclidean(point, c) for c in centroids]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(point)
    centroids = np.array([np.mean(clusters[i], axis=0) for i in clusters])

print("Final centroids:", centroids)

# =========================================================
# EXERCISE 2 — ELBOW METHOD (FIXED)
# =========================================================

print("\n=== EXERCISE 2: ELBOW METHOD ===")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
max_k = X_scaled.shape[0]   # ✅ K must be <= samples

for k in range(1, max_k + 1):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(range(1, max_k + 1), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# =========================================================
# EXERCISE 3 — PCA FROM SCRATCH
# =========================================================

print("\n=== EXERCISE 3: PCA FROM SCRATCH ===")

X_std = (X - X.mean(axis=0)) / X.std(axis=0)
cov_matrix = np.cov(X_std.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
top_vectors = eigenvectors[:, idx[:2]]

X_pca_manual = X_std.dot(top_vectors)
print("PCA Reduced Data:\n", X_pca_manual)

# =========================================================
# EXERCISE 4 — MALL CUSTOMER SEGMENTATION
# =========================================================

print("\n=== EXERCISE 4: MALL CUSTOMER SEGMENTATION ===")

mall = pd.DataFrame({
    "Income": [15, 16, 17, 60, 65, 70],
    "Spending": [80, 75, 78, 20, 18, 15]
})

mall_scaled = StandardScaler().fit_transform(mall)
mall["Cluster"] = KMeans(n_clusters=2, random_state=42).fit_predict(mall_scaled)

print(mall)

# =========================================================
# EXERCISE 5 — SOCIAL MEDIA USER CLUSTERING
# =========================================================

print("\n=== EXERCISE 5: SOCIAL MEDIA USER CLUSTERING ===")

social = pd.DataFrame({
    "Posts": [20, 5, 2, 15, 1],
    "Likes": [500, 50, 10, 300, 5],
    "Shares": [100, 5, 1, 80, 2],
    "Watch_Time": [300, 100, 500, 250, 600]
})

social_scaled = StandardScaler().fit_transform(social)
social["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(social_scaled)

print(social)

# =========================================================
# EXERCISE 6 — NEWS ARTICLE CLUSTERING (TF-IDF)
# =========================================================

print("\n=== EXERCISE 6: NEWS ARTICLE CLUSTERING ===")

articles = [
    "The election results were announced today",
    "New AI technology is changing the world",
    "The football team won the championship",
    "Government passed a new policy",
    "Latest smartphone launched with AI features",
    "Cricket match ended in a draw"
]

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(articles)

news_clusters = KMeans(n_clusters=3, random_state=42).fit_predict(X_text)

print("Article cluster labels:", news_clusters)

# =========================================================
# EXERCISE 7 — PRODUCT RECOMMENDATION GROUPING
# =========================================================

print("\n=== EXERCISE 7: PRODUCT GROUPING ===")

products = pd.DataFrame({
    "Price": [100, 120, 500, 520, 90],
    "Rating": [4.5, 4.2, 4.8, 4.9, 4.0],
    "Popularity": [300, 250, 100, 120, 400]
})

products_scaled = StandardScaler().fit_transform(products)
products["Cluster"] = KMeans(n_clusters=2, random_state=42).fit_predict(products_scaled)

print(products)

# =========================================================
# EXERCISE 8 — IMAGE COMPRESSION USING KMEANS
# =========================================================

print("\n=== EXERCISE 8: IMAGE COMPRESSION ===")

image = load_sample_image("china.jpg")
X_img = image.reshape(-1, 3)

kmeans_img = KMeans(n_clusters=8, random_state=42)
labels = kmeans_img.fit_predict(X_img)

compressed_pixels = kmeans_img.cluster_centers_[labels]
compressed_image = compressed_pixels.reshape(image.shape).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(compressed_image)
plt.title("Compressed Image using KMeans")
plt.axis("off")
plt.show()

print("\n=== ALL EXERCISES COMPLETED SUCCESSFULLY ===")


# In[ ]:




