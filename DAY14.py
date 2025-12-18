#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================================
# PART 7 — MINI CHALLENGE (MANDATORY)
# Neural Network on MNIST with Justification
# ==========================================================

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -----------------------------
# Step 2: Load MNIST Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# -----------------------------
# Step 3: Build Neural Network
# -----------------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

# -----------------------------
# Step 4: Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show architecture
model.summary()

# -----------------------------
# Step 5: Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# -----------------------------
# Step 6: Plot Training Graphs
# -----------------------------
plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

plt.show()

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", test_accuracy)

# -----------------------------
# Step 8: Explanation (Print)
# -----------------------------
print("\nMODEL DECISIONS & TRADE-OFFS:\n")

print("1. ReLU activation avoids vanishing gradients and speeds up learning.")
print("2. Softmax is used for multi-class digit classification (0–9).")
print("3. Batch size of 32 balances speed and stable gradient updates.")
print("4. 10 epochs allow convergence without overfitting.")
print("5. Adam optimizer adapts learning rates automatically.\n")

print("TRADE-OFFS:")
print("- Larger batch sizes train faster but may generalize worse.")
print("- More epochs improve accuracy but risk overfitting.")
print("- Deeper networks improve performance but increase computation cost.")


# In[7]:


# ==========================================================
# CODING TASK 1: SIMPLE NN WITH DIFFERENT ACTIVATIONS
# Dataset: MNIST
# ==========================================================

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import time

# -----------------------------
# Load and preprocess dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# -----------------------------
# Function to build & train model
# -----------------------------
def train_model(activation_name):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation_name),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    end_time = time.time()

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return round(test_accuracy, 4), round(end_time - start_time, 2)

# -----------------------------
# Train models with activations
# -----------------------------
activations = ["sigmoid", "tanh", "relu"]
results = []

for act in activations:
    acc, t = train_model(act)
    results.append((act.upper(), acc, f"{t} sec"))

# -----------------------------
# Print comparison table
# -----------------------------
print("\nACTIVATION FUNCTION COMPARISON\n")
print("Activation | Accuracy | Training Time")
print("-------------------------------------")
for r in results:
    print(f"{r[0]:<10} | {r[1]:<8} | {r[2]}")

# -----------------------------
# Analysis Answers
# -----------------------------
print("\nANALYSIS ANSWERS\n")

print("1. Which activation converges faster?")
print("→ ReLU converges faster because it avoids vanishing gradients.\n")

print("2. Which activation struggles?")
print("→ Sigmoid struggles due to gradient saturation.\n")

print("3. Why does sigmoid perform poorly in deep networks?")
print("→ Sigmoid compresses values between 0 and 1,")
print("  causing vanishing gradients and slow learning.\n")

# -----------------------------
# Real-World Mapping
# -----------------------------
print("REAL-WORLD ACTIVATION USAGE\n")
print("Sigmoid → Binary decisions (approve/reject)")
print("ReLU    → Vision, Speech, NLP")
print("Tanh    → RNNs, Time-Series")


# In[12]:


# ==========================================================
# CODING TASK 2: EPOCH EXPERIMENT (LEARNING BEHAVIOR)
# Dataset: MNIST
# ==========================================================

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -----------------------------
# Load and preprocess dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# -----------------------------
# Function to train model
# -----------------------------
def train_with_epochs(num_epochs):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    return history

# -----------------------------
# Train models with different epochs
# -----------------------------
history_5 = train_with_epochs(5)
history_10 = train_with_epochs(10)
history_20 = train_with_epochs(20)

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_5.history["accuracy"], label="Train Acc (5 epochs)")
plt.plot(history_5.history["val_accuracy"], label="Val Acc (5 epochs)")

plt.plot(history_10.history["accuracy"], label="Train Acc (10 epochs)")
plt.plot(history_10.history["val_accuracy"], label="Val Acc (10 epochs)")

plt.plot(history_20.history["accuracy"], label="Train Acc (20 epochs)")
plt.plot(history_20.history["val_accuracy"], label="Val Acc (20 epochs)")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")
plt.legend()

# -----------------------------
# Plot Loss
# -----------------------------
plt.subplot(1, 2, 2)
plt.plot(history_5.history["loss"], label="Train Loss (5 epochs)")
plt.plot(history_5.history["val_loss"], label="Val Loss (5 epochs)")

plt.plot(history_10.history["loss"], label="Train Loss (10 epochs)")
plt.plot(history_10.history["val_loss"], label="Val Loss (10 epochs)")

plt.plot(history_20.history["loss"], label="Train Loss (20 epochs)")
plt.plot(history_20.history["val_loss"], label="Val Loss (20 epochs)")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()

plt.show()

# -----------------------------
# Print Final Metrics
# -----------------------------
print("\nFINAL RESULTS\n")

print("5 Epochs:")
print("Train Accuracy:", history_5.history["accuracy"][-1])
print("Validation Accuracy:", history_5.history["val_accuracy"][-1])
print("Training Loss:", history_5.history["loss"][-1])

print("\n10 Epochs:")
print("Train Accuracy:", history_10.history["accuracy"][-1])
print("Validation Accuracy:", history_10.history["val_accuracy"][-1])
print("Training Loss:", history_10.history["loss"][-1])

print("\n20 Epochs:")
print("Train Accuracy:", history_20.history["accuracy"][-1])
print("Validation Accuracy:", history_20.history["val_accuracy"][-1])
print("Training Loss:", history_20.history["loss"][-1])

# -----------------------------
# Analysis Answers
# -----------------------------
print("\nANALYSIS ANSWERS\n")

print("1. At which epoch does validation accuracy stop improving?")
print("→ Around 10 epochs, validation accuracy plateaus.\n")

print("2. Does loss keep decreasing while accuracy stagnates?")
print("→ Yes, training loss keeps decreasing even when validation accuracy stops improving.\n")

print("3. Identify overfitting point:")
print("→ After ~10 epochs, training accuracy improves but validation accuracy stagnates or drops.\n")

# -----------------------------
# Real-World Analogy
# -----------------------------
print("REAL-WORLD ANALOGY\n")
print("• Too few epochs → Like studying too little → poor understanding")
print("• Optimal epochs → Balanced learning → good exam performance")
print("• Too many epochs → Memorization & burnout → overfitting")


# In[ ]:





# In[ ]:




