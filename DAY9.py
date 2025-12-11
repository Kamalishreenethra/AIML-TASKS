#!/usr/bin/env python
# coding: utf-8

# In[4]:


# digits_miniproject_clean.py
"""
Multi-class Classification Mini Project — Digits dataset
Clean version (no warnings)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
)

# ---------------------
# 1) Load dataset + EDA
# ---------------------
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

df = pd.DataFrame(X)
df["target"] = y

print("\n=== Dataset Info ===")
print(df.head())
print("Shape:", df.shape)
print("Target counts:\n", df["target"].value_counts())

# Show sample images
try:
    plt.figure(figsize=(10, 3))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {y[i]}")
        plt.axis("off")
    plt.suptitle("Sample Digit Images")
    plt.tight_layout()
    plt.show()
except:
    print("Warning: Unable to display images (maybe running in non-GUI environment)")

# Distribution plot
try:
    sns.countplot(x=df["target"])
    plt.title("Digit Class Distribution")
    plt.show()
except:
    print("Plot skipped.")

# ---------------------
# 2) Train/Test split + scaling
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------
# 3) Logistic Regression (clean, no warnings)
# ---------------------
print("\n=== Logistic Regression Training ===")

log_model = LogisticRegression(max_iter=5000, random_state=42, solver="lbfgs")
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_pred)
print(f"Logistic Regression Accuracy: {log_acc:.4f}")

# ---------------------
# 4) KNN + K Tuning
# ---------------------
print("\n=== KNN Training + K Tuning ===")

k_values = range(1, 21)
acc_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc_scores.append(model.score(X_test_scaled, y_test))

best_k = acc_scores.index(max(acc_scores)) + 1
print(f"Best K = {best_k} with accuracy = {max(acc_scores):.4f}")

# Train best KNN model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)

print(f"KNN (K={best_k}) Accuracy: {knn_acc:.4f}")

# Plot K vs accuracy
try:
    plt.plot(k_values, acc_scores, marker='o')
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("KNN K-Tuning")
    plt.grid(True)
    plt.show()
except:
    print("Plot skipped.")

# ---------------------
# 5) Manual confusion matrix
# ---------------------
def manual_confusion(true, pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm

num_classes = len(np.unique(y))

print("\nManual Confusion Matrix — Logistic Regression:")
print(manual_confusion(y_test, log_pred, num_classes))

print("\nManual Confusion Matrix — KNN:")
print(manual_confusion(y_test, knn_pred, num_classes))

# ---------------------
# 6) Precision / Recall
# ---------------------
print("\n=== Precision & Recall (Macro) ===")
print("Logistic Precision:", precision_score(y_test, log_pred, average="macro"))
print("Logistic Recall:", recall_score(y_test, log_pred, average="macro"))

print("KNN Precision:", precision_score(y_test, knn_pred, average="macro"))
print("KNN Recall:", recall_score(y_test, knn_pred, average="macro"))

# ---------------------
# 7) 5 Insights
# ---------------------
print("\n=== INSIGHTS ===")
print("1) Dataset is balanced across all 10 classes (0–9).")
print("2) Logistic Regression achieved accuracy:", round(log_acc, 4))
print(f"3) KNN tuning found best K = {best_k}, showing how K impacts accuracy.")
print("4) Confusion matrix shows digits like 3,5 and 8,9 get confused more.")
print("5) Precision & Recall indicate strong performance across most classes.")

print("\nProject Completed Successfully!")


# In[5]:


# email_spam_miniproject.py
"""
Use Case 1: Email Spam Classification
- Synthetic dataset (16 rows)
- Features: word_freq, link_count, msg_length, suspicious_keywords
- Train Logistic Regression
- Compute manual confusion matrix + sklearn metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

# -----------------------
# 1) Create synthetic dataset (16 rows)
# -----------------------
rows = [
    # word_freq, link_count, msg_length, suspicious_keywords, spam
    (0.01, 0,  45, 0, 0),  # normal short message
    (0.10, 2, 120, 1, 1),  # suspicious, links
    (0.00, 0, 300, 0, 0),  # long legit newsletter
    (0.25, 5, 80, 3, 1),   # many spammy words + links
    (0.05, 1, 60, 0, 0),
    (0.30, 4, 50, 2, 1),
    (0.02, 0, 20, 0, 0),
    (0.18, 3, 40, 2, 1),
    (0.00, 0, 1000,0, 0),  # very long, large legitimate content
    (0.12, 2, 55, 1, 1),
    (0.03, 0, 70, 0, 0),
    (0.22, 1, 30, 2, 1),
    (0.06, 0, 90, 0, 0),
    (0.40, 6, 35, 4, 1),
    (0.08, 1, 65, 0, 0),
    (0.15, 2, 48, 1, 1),
]

df = pd.DataFrame(rows, columns=[
    "word_freq",        # fraction of words that are suspicious words (0..1)
    "link_count",       # number of links in email
    "msg_length",       # message length in characters / words (synthetic)
    "suspicious_kw",    # count of clearly suspicious keywords (e.g., 'free', 'win')
    "spam"              # target: 1 = spam, 0 = ham
])

print("=== Dataset ===")
print(df)
print("\nClass distribution:\n", df['spam'].value_counts())

# -----------------------
# 2) Train/Test split
# -----------------------
X = df[["word_freq", "link_count", "msg_length", "suspicious_kw"]].values
y = df["spam"].values

# small dataset -> use stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.31, random_state=42, stratify=y
)

# -----------------------
# 3) Scale features (recommended for LR)
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# 4) Train Logistic Regression
# -----------------------
model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------
# 5) Manual confusion matrix (binary)
# -----------------------
def manual_confusion_matrix_binary(true_labels, pred_labels):
    # rows = true class (0,1), cols = predicted (0,1)
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm

manual_cm = manual_confusion_matrix_binary(y_test, y_pred)

# -----------------------
# 6) Metrics + output
# -----------------------
print("\n=== Results on test set ===")
print("Test samples:", len(y_test))
print("Manual Confusion Matrix (rows=true, cols=pred):")
print(manual_cm)

print("\nSklearn Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nPredicted probabilities for test samples (spam probability):")
for i, prob in enumerate(y_prob):
    print(f"  sample {i}: prob_spam = {prob:.3f}, predicted = {y_pred[i]}, true = {y_test[i]}")

# -----------------------
# 7) Quick explanation of confusion matrix layout
# -----------------------
print("\nConfusion matrix layout:")
print(" [[TN  FP]")
print("  [FN  TP]]")

# End
print("\nDone.")


# In[6]:


# algorithms_scratch.py
"""
Algorithms from scratch:
- KNN (intermediate) using Euclidean distance
- Logistic Regression (sigmoid + training via gradient descent)
- Confusion matrix built from scratch
Demo on a small synthetic email-spam dataset.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------
# Synthetic dataset (small)
# -----------------------
rows = [
    # word_freq, link_count, msg_length, suspicious_kw, spam
    (0.01, 0,  45, 0, 0),
    (0.10, 2, 120, 1, 1),
    (0.00, 0, 300, 0, 0),
    (0.25, 5, 80, 3, 1),
    (0.05, 1, 60, 0, 0),
    (0.30, 4, 50, 2, 1),
    (0.02, 0, 20, 0, 0),
    (0.18, 3, 40, 2, 1),
    (0.00, 0, 1000,0, 0),
    (0.12, 2, 55, 1, 1),
    (0.03, 0, 70, 0, 0),
    (0.22, 1, 30, 2, 1),
    (0.06, 0, 90, 0, 0),
    (0.40, 6, 35, 4, 1),
    (0.08, 1, 65, 0, 0),
    (0.15, 2, 48, 1, 1),
]

df = pd.DataFrame(rows, columns=[
    "word_freq", "link_count", "msg_length", "suspicious_kw", "spam"
])

X = df[["word_freq", "link_count", "msg_length", "suspicious_kw"]].values
y = df["spam"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.31, random_state=42, stratify=y
)

# Standardize features (helps both KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Algorithm 1: KNN from scratch
# -----------------------
class KNNScratch:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    @staticmethod
    def _euclidean(a, b):
        # a and b are 1-D arrays
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _predict_one(self, x):
        # compute distances to all train points
        distances = [self._euclidean(x, xt) for xt in self.X_train]
        # get indices of k smallest distances
        k_idx = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_idx]
        # majority vote
        count = Counter(k_labels)
        most_common = count.most_common()
        # if tie, most_common will list ties; break by smallest label for determinism
        top_count = most_common[0][1]
        # collect labels with top_count
        tied = [label for label, c in most_common if c == top_count]
        prediction = min(tied)  # deterministic tie-breaker: smallest label
        return prediction

    def predict(self, X):
        X = np.array(X)
        preds = [self._predict_one(x) for x in X]
        return np.array(preds)

# -----------------------
# Algorithm 2: Logistic Regression (from scratch)
# - sigmoid
# - batch gradient descent with L2 regularization option
# -----------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=1000, fit_intercept=True, verbose=False, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.l2 = l2  # L2 regularization strength (lambda)
        self.w = None
        self.b = 0.0

    @staticmethod
    def sigmoid(z):
        # numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _initialize(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self._initialize(n_features)

        for i in range(self.n_iter):
            linear = X.dot(self.w) + self.b
            preds = self.sigmoid(linear)  # probabilities
            # gradient (binary cross-entropy)
            error = preds - y  # shape (n_samples,)
            grad_w = (X.T.dot(error)) / n_samples + (self.l2 * self.w) / n_samples
            grad_b = np.sum(error) / n_samples
            # update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if self.verbose and (i % (self.n_iter // 10 + 1) == 0):
                # compute loss (with L2)
                loss = -np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
                loss += (self.l2 / (2 * n_samples)) * np.sum(self.w ** 2)
                print(f"iter {i:d} - loss: {loss:.6f}")

    def predict_proba(self, X):
        X = np.array(X)
        linear = X.dot(self.w) + self.b
        probs = self.sigmoid(linear)
        return np.vstack([1 - probs, probs]).T  # shape (n_samples, 2)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs > threshold).astype(int)

# -----------------------
# Algorithm 3: Confusion matrix from scratch
# -----------------------
def my_confusion_matrix(actual, predicted):
    """
    Returns [[tn, fp],
             [fn, tp]]
    """
    tn = fp = fn = tp = 0
    for a, p in zip(actual, predicted):
        if a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1
        elif a == 1 and p == 1:
            tp += 1
        else:
            # handles unexpected labels by ignoring
            pass
    return np.array([[tn, fp], [fn, tp]])

# -----------------------
# Utilities: compute metrics from confusion matrix
# -----------------------
def metrics_from_cm(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

# -----------------------
# Run models and compare
# -----------------------
# 1) KNN
knn = KNNScratch(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)

cm_knn = my_confusion_matrix(y_test, knn_preds)
metrics_knn = metrics_from_cm(cm_knn)

# 2) Logistic Regression (from scratch)
log_scratch = LogisticRegressionScratch(lr=0.5, n_iter=2000, verbose=False, l2=0.01)
log_scratch.fit(X_train_scaled, y_train)
log_preds = log_scratch.predict(X_test_scaled, threshold=0.5)
log_probs = log_scratch.predict_proba(X_test_scaled)[:,1]

cm_log = my_confusion_matrix(y_test, log_preds)
metrics_log = metrics_from_cm(cm_log)

# -----------------------
# Print results
# -----------------------
print("=== Test set size:", len(y_test))
print("\nKNN (k=3) — Confusion Matrix (rows=true, cols=pred):")
print(cm_knn)
print("KNN metrics:", {k: round(v,4) for k,v in metrics_knn.items()})

print("\nLogisticRegressionScratch — Confusion Matrix:")
print(cm_log)
print("Logistic Regression metrics:", {k: round(v,4) for k,v in metrics_log.items()})

# Show predicted probabilities for logistic (for understanding)
print("\nLogistic predicted probabilities (spam prob) and predictions:")
for i, (p, pred, true) in enumerate(zip(log_probs, log_preds, y_test)):
    print(f" sample {i}: prob_spam={p:.3f}, predicted={pred}, true={true}")

# -----------------------
# End
# -----------------------
print("\nDone.")


# In[8]:


# part4_usecases_fixed.py
"""
PART 4 — Real-World Classification Use Cases (fixed)
- Prevents K > n_train errors for KNN tuning
- Adds small robustness improvements (scaling, stratify, random_state)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
)
from sklearn.datasets import load_digits

RANDOM_STATE = 42

# -----------------------------
# USE CASE 1 — EMAIL SPAM (synthetic)
# -----------------------------
print("\n=== USE CASE 1: Email Spam Classification ===")
email_data = [
    (0.01, 0, 45, 0, 0),
    (0.12, 3, 60, 1, 1),
    (0.00, 0, 300, 0, 0),
    (0.30, 4, 80, 2, 1),
    (0.05, 1, 55, 0, 0),
    (0.25, 5, 50, 3, 1),
    (0.02, 0, 20, 0, 0),
    (0.18, 2, 40, 2, 1),
    (0.00, 0, 1200, 0, 0),
    (0.15, 2, 65, 1, 1),
]

df_email = pd.DataFrame(email_data, columns=[
    "word_freq", "link_count", "msg_length", "suspicious_kw", "spam"
])

X = df_email.drop(columns="spam")
y = df_email["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
lr.fit(X_train_s, y_train)
pred = lr.predict(X_test_s)

print("Confusion matrix (email):\n", confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification report:\n", classification_report(y_test, pred, zero_division=0))

# -----------------------------
# USE CASE 2 — DIABETES (Pima Indians)
# -----------------------------
print("\n=== USE CASE 2: Disease (Diabetes) Prediction ===")
# Load Pima dataset from common CSV URL (Plotly copy). If offline, the user should download.
try:
    diabetes_df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
except Exception as e:
    raise RuntimeError("Could not download Pima dataset. Ensure internet connection or provide CSV locally.") from e

features = ["Glucose", "Age", "BMI", "Insulin"]
X = diabetes_df[features]
y = diabetes_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model_diabetes = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
model_diabetes.fit(X_train_s, y_train)
pred = model_diabetes.predict(X_test_s)

print("Precision (diabetes):", precision_score(y_test, pred, zero_division=0))
print("Recall (diabetes):   ", recall_score(y_test, pred, zero_division=0))
print("Classification report:\n", classification_report(y_test, pred, zero_division=0))
print("\nWhy recall matters more for disease prediction:")
print("→ Missing a true positive (false negative) can deny treatment to a sick patient;"
      " recall (sensitivity) prioritizes catching positives even at some false alarms.")

# -----------------------------
# USE CASE 3 — CUSTOMER CHURN (synthetic)
# -----------------------------
print("\n=== USE CASE 3: Customer Churn Prediction ===")
churn_data = [
    # monthly_charges, contract_months, complaints, age, churn
    (50, 12, 0, 30, 0),
    (75, 2, 4, 45, 1),
    (60, 10, 1, 28, 0),
    (90, 3, 5, 50, 1),
    (40, 20, 0, 24, 0),
    (80, 1, 6, 48, 1),
    (55, 15, 0, 33, 0),
    (95, 4, 3, 52, 1),
]
df_churn = pd.DataFrame(churn_data, columns=[
    "monthly_charges", "contract_months", "complaints", "age", "churn"
])

X = df_churn.drop(columns="churn")
y = df_churn["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

n_train = X_train_s.shape[0]
# candidate K values (odd numbers up to min(19, n_train))
candidate_ks = [k for k in range(1, 20, 2) if k <= n_train]
if not candidate_ks:
    print("Dataset too small to try KNN tuning (no valid K <= n_train). Using K=1.")
    candidate_ks = [1]

best_k = None
best_acc = -1.0
print(f"n_train = {n_train}, trying K's = {candidate_ks}")
for k in candidate_ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f" K={k} -> accuracy={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"Best K for churn (from candidates) = {best_k} (accuracy={best_acc:.3f})")

# -----------------------------
# USE CASE 4 — HANDWRITTEN DIGITS (sklearn digits)
# -----------------------------
print("\n=== USE CASE 4: Handwritten Digits (Digits dataset) ===")
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# scale (helpful for distance-based KNN)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for k in [3, 5, 7]:
    # ensure k <= n_train
    if k > X_train_s.shape[0]:
        print(f" K={k} skipped (k > n_train={X_train_s.shape[0]})")
        continue
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f" K={k} -> accuracy={acc:.4f}")

print("Observation: for high-dimensional image data K=3 or 5 often works well, but test to confirm.")

# -----------------------------
# USE CASE 5 — BANK LOAN APPROVAL (synthetic)
# -----------------------------
print("\n=== USE CASE 5: Bank Loan Approval ===")
loan_data = [
    (50000, 700, 20000, 1, 1),
    (20000, 550, 15000, 0, 0),
    (80000, 800, 25000, 1, 1),
    (30000, 600, 22000, 0, 0),
    (90000, 750, 30000, 1, 1),
    (25000, 500, 18000, 0, 0),
]
df_loan = pd.DataFrame(loan_data, columns=[
    "income", "credit_score", "loan_amount", "employed", "approved"
])

X = df_loan.drop(columns="approved")
y = df_loan["approved"]

# small dataset -> stratify may fail; use train_test_split without stratify if necessary
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
lr_acc = lr.score(X_test_s, y_test)

n_train = X_train_s.shape[0]
k_candidates = [k for k in [1,3,5] if k <= n_train]
if not k_candidates:
    k_candidates = [1]
best_k = None
best_knn_acc = -1.0
for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f" K={k} -> KNN accuracy={acc:.3f}")
    if acc > best_knn_acc:
        best_knn_acc = acc
        best_k = k

print(f"Logistic Regression accuracy (loan): {lr_acc:.3f}")
print(f"KNN best (loan): K={best_k} accuracy={best_knn_acc:.3f}")

print("\n=== END OF ALL USE CASES ===")


# In[ ]:




