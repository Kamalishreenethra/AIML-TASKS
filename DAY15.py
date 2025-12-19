#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.getcwd())


# In[ ]:


get_ipython().run_line_magic('pip', 'install scipy')


# In[ ]:


import os

folders = [
    r"C:\Users\HP\data\train\food",
    r"C:\Users\HP\data\train\non_food",
    r"C:\Users\HP\data\test\food",
    r"C:\Users\HP\data\test\non_food"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ All dataset folders created successfully")


# In[ ]:


print(os.listdir(r"C:\Users\HP\data"))
print(os.listdir(r"C:\Users\HP\data\train"))


# In[ ]:


C:\Users\HP\data\train\food        → Food images
C:\Users\HP\data\train\non_food    → Non-food images
C:\Users\HP\data\test\food         → Food images
C:\Users\HP\data\test\non_food     → Non-food images


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -----------------------------
# Image parameters
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 8

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------
# Load images
# -----------------------------
train_data = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    "data/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# -----------------------------
# Plot Accuracy & Loss
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.show()

# -----------------------------
# Save model
# -----------------------------
model.save("food_vs_nonfood_cnn.h5")
print("Model saved successfully!")


# In[ ]:


import os

print(os.path.exists("data/train"))
print(os.listdir("data") if os.path.exists("data") else "data folder missing")


# In[ ]:


"data/train"


# In[ ]:


# ================================
# FACE MASK DETECTOR - FULL CODE
# ================================

# -------- STEP 1: IMPORT LIBRARIES --------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# -------- STEP 2: PARAMETERS --------
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10

# -------- STEP 3: DATA AUGMENTATION --------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# -------- STEP 4: LOAD DATASET (✔ CORRECT PATH) --------
train_data = train_datagen.flow_from_directory(
    r"C:\Users\HP\face_mask_dataset\data\train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    r"C:\Users\HP\face_mask_dataset\data\test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# -------- STEP 5: BUILD CNN MODEL --------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# -------- STEP 6: COMPILE MODEL --------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------- STEP 7: TRAIN MODEL --------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# -------- STEP 8: EVALUATION --------
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_data.classes

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, target_names=["With Mask", "Without Mask"]))

print("CONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

# -------- STEP 9: TEST SINGLE IMAGE --------
from tensorflow.keras.preprocessing import image

img = image.load_img("test_face.jpg", target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("😷 MASK DETECTED")
else:
    print("❌ NO MASK")

# -------- STEP 10: SAVE MODEL --------
model.save("face_mask_detector.h5")
print("✅ Model saved successfully")


# In[ ]:


get_ipython().run_line_magic('pip', 'install scikit-learn')


# In[ ]:


# ============================================================
# HANDWRITTEN ALPHABET RECOGNIZER (A–Z)
# Dataset: EMNIST Letters
# Model: CNN
# Tasks:
# 1. Load dataset
# 2. Train CNN
# 3. Visualize feature maps
# 4. Predict custom handwritten letter
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

# -----------------------------
# 2. LOAD EMNIST DATASET
# -----------------------------
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully!")
print(ds_info)

# -----------------------------
# 3. PREPROCESSING
# -----------------------------
IMG_SIZE = 28
BATCH_SIZE = 128

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))
    label = label - 1   # Convert 1–26 → 0–25
    return image, label

train_ds = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
test_ds  = ds_test.map(preprocess).batch(BATCH_SIZE)

# -----------------------------
# 4. BUILD CNN MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(26, activation="softmax")  # A–Z
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# -----------------------------
# 6. PLOT TRAINING CURVES
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()

# -----------------------------
# 7. VISUALIZE FEATURE MAPS
# -----------------------------
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)

for images, labels in test_ds.take(1):
    sample_image = images[0:1]

feature_maps = feature_extractor.predict(sample_image)

plt.figure(figsize=(12,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(feature_maps[0,:,:,i], cmap="gray")
    plt.axis("off")

plt.suptitle("CNN Feature Maps (Edges & Strokes)")
plt.show()

# -----------------------------
# 8. PREDICT CUSTOM HANDWRITTEN LETTER
# -----------------------------
def predict_letter(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28,28))
    img = np.array(img)

    # Invert image (EMNIST style)
    img = 255 - img

    img = img / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    letter_index = np.argmax(prediction)
    letter = chr(letter_index + ord("A"))

    plt.imshow(img.reshape(28,28), cmap="gray")
    plt.title(f"Predicted Letter: {letter}")
    plt.axis("off")
    plt.show()

# -----------------------------
# 9. TEST WITH YOUR IMAGE
# -----------------------------
# Put a 28x28 handwritten letter image path below
# Example: predict_letter("A.png")

# predict_letter("your_letter.png")


# In[ ]:




