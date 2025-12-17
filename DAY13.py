#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ----------------------------------------
# Task 1 — Build a Neural Network on MNIST
# ----------------------------------------

# Step 1 — Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2 — Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 3 — Preprocess the data

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images (28×28 → 784)
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

print("Flattened training shape:", X_train.shape)
print("Flattened testing shape:", X_test.shape)

# Step 4 — Build the Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Step 5 — Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model architecture
model.summary()

# Step 6 — Train the model
model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.1
)

# Step 7 — Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy)


# In[3]:


# ============================================
# Practice Exercise 7 — Model Experiments
# Topic: MNIST Neural Network Experiments
# ============================================

# Step 1 — Import libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Step 2 — Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 3 — Normalize data (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 4 — One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ============================================
# 🔬 EXPERIMENT SETTINGS (CHANGE THESE)
# ============================================

NEURONS = 32            # Try: 32, 64, 128, 256
ACTIVATION = "tanh"      # Try: "relu", "tanh", "sigmoid"
EPOCHS = 5              # Try: 5, 10, 20
BATCH_SIZE = 32

print("Model Configuration:")
print(f"Neurons       : {NEURONS}")
print(f"Activation    : {ACTIVATION}")
print(f"Epochs        : {EPOCHS}")
print("=" * 40)

# ============================================
# Step 5 — Build the Neural Network
# ============================================

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(NEURONS, activation=ACTIVATION),
    Dense(NEURONS // 2, activation=ACTIVATION),
    Dense(10, activation="softmax")
])

# Step 6 — Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 7 — Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# Step 8 — Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nFinal Test Accuracy:", round(test_accuracy * 100, 2), "%")


# In[ ]:


# ============================================
# PART 8 — DRAW A DIGIT & PREDICT 🤖
# FULLY CORRECTED & STUDENT-SAFE VERSION
# ============================================

# Step 1 — Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tkinter import Tk, filedialog

# ============================================
# Step 2 — Load and prepare MNIST data
# ============================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ============================================
# Step 3 — Build & train the model
# ============================================

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training the model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# ============================================
# Step 4 — Select digit image using file picker
# ============================================

Tk().withdraw()  # Hide Tkinter main window

IMAGE_PATH = filedialog.askopenfilename(
    title="Select a digit image (0–9)",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not IMAGE_PATH:
    raise FileNotFoundError("No image selected!")

# ============================================
# Step 5 — Load & preprocess image
# ============================================

# Open image and convert to grayscale
img = Image.open(IMAGE_PATH).convert("L")

# Resize to 28x28
img = img.resize((28, 28))

# Convert to numpy array
img_array = np.array(img)

# Invert colors if background is white
img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0

# Reshape for model input
img_array = img_array.reshape(1, 28, 28)

# ============================================
# Step 6 — Display input image
# ============================================

plt.imshow(img_array[0], cmap="gray")
plt.title("Input Digit")
plt.axis("off")
plt.show()

# ============================================
# Step 7 — Predict digit
# ============================================

prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

print("🤖 Model Prediction:", predicted_digit)
print("Prediction Probabilities:", np.round(prediction[0], 3))


# In[1]:


# ============================================
# PRACTICE EXERCISE 8 — MODEL FAILURE ANALYSIS
# ============================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# --------------------------------------------
# Step 1 — Load MNIST data
# --------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# --------------------------------------------
# Step 2 — Build & train model
# --------------------------------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=1)

# --------------------------------------------
# Step 3 — Predict on test data
# --------------------------------------------
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# --------------------------------------------
# Step 4 — Find misclassified digits
# --------------------------------------------
misclassified_idx = np.where(predicted_labels != y_test)[0]

print("Total test samples:", len(X_test))
print("Misclassified samples:", len(misclassified_idx))

# --------------------------------------------
# Step 5 — Display some misclassified digits
# --------------------------------------------
plt.figure(figsize=(10, 5))

for i, idx in enumerate(misclassified_idx[:8]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(f"True: {y_test[idx]} | Pred: {predicted_labels[idx]}")
    plt.axis("off")

plt.suptitle("Misclassified Digits (Wrong Predictions)")
plt.show()


# In[2]:


# ============================================
# Algorithm 1 — Single Neuron From Scratch
# ============================================

import math

# --------------------------------------------
# Step 1 — Initialize weights & bias
# --------------------------------------------
w = 0.5        # weight
b = 0.0        # bias
lr = 0.1       # learning rate

# Input and true output
x = 1.0        # input feature
y = 1.0        # true label

print("Initial weight:", w)
print("Initial bias:", b)

# --------------------------------------------
# Step 2 — Compute z = wx + b
# --------------------------------------------
z = w * x + b

# --------------------------------------------
# Step 3 — Apply sigmoid activation
# --------------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

y_pred = sigmoid(z)

print("Predicted output:", round(y_pred, 4))

# --------------------------------------------
# Step 4 — Compute loss (Mean Squared Error)
# --------------------------------------------
loss = (y - y_pred) ** 2
print("Loss:", round(loss, 4))

# --------------------------------------------
# Step 5 — Manual weight & bias update
# --------------------------------------------

# Derivatives (chain rule)
dL_dy = -2 * (y - y_pred)
dy_dz = y_pred * (1 - y_pred)
dz_dw = x
dz_db = 1

# Gradients
dL_dw = dL_dy * dy_dz * dz_dw
dL_db = dL_dy * dy_dz * dz_db

# Update
w = w - lr * dL_dw
b = b - lr * dL_db

print("\nUpdated weight:", round(w, 4))
print("Updated bias:", round(b, 4))


# In[4]:


# ============================================
# MINI PROJECT — DIGIT CLASSIFIER
# ============================================

# Step 1 — Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# ============================================
# Step 2 — Load MNIST dataset
# ============================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# ============================================
# Step 3 — Build Neural Network
# ============================================

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================
# Step 4 — Train the model
# ============================================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=10,          # 🔧 Tune epochs
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================================
# Step 5 — Evaluate the model
# ============================================

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# ============================================
# Step 6 — Tune Parameters (Experiment)
# ============================================

print("\nTry tuning:")
print("- Neurons: 64, 128, 256")
print("- Epochs: 5, 10, 20")
print("- Activation: relu / tanh")

# ============================================
# Step 7 — Test with Custom Handwritten Digit
# ============================================

print("\nCurrent folder:", os.getcwd())
print("Files:", os.listdir())

IMAGE_PATH = "digit.png"

img = Image.open(IMAGE_PATH).convert("L")
img = img.resize((28, 28))

img_array = np.array(img)

# Fix background issues
if img_array.mean() > 127:
    img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28)

# Show input image
plt.imshow(img_array[0], cmap="gray")
plt.title("Custom Input Digit")
plt.axis("off")
plt.show()

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

print("🤖 Predicted Digit:", predicted_digit)
print("Prediction Probabilities:", np.round(prediction[0], 3))


# In[ ]:




