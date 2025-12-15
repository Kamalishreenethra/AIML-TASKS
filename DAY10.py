#!/usr/bin/env python
# coding: utf-8

# In[5]:


print("🍴 What to Eat Today? — Decision Tree Game 🍴\n")

hungry = input("Are you hungry? (yes/no): ").lower()

if hungry == "yes":
    spicy = input("Do you want something spicy? (yes/no): ").lower()

    if spicy == "yes":
        print("\n✅ Decision: Eat Biryani 🍛")

    else:
        vegetarian = input("Do you want vegetarian food? (yes/no): ").lower()

        if vegetarian == "yes":
            print("\n✅ Decision: Eat Salad 🥗")
        else:
            print("\n✅ Decision: Eat Burger 🍔")

else:
    print("\n✅ Decision: Have Tea or Coffee ☕")


# In[6]:


import math

# -----------------------------
# Parent node data
# -----------------------------
yes = 7
no = 3
total = yes + no

p_yes = yes / total
p_no = no / total

# -----------------------------
# Gini Impurity
# -----------------------------
gini = 1 - (p_yes ** 2 + p_no ** 2)

print("Gini Impurity (Parent):", round(gini, 3))

# -----------------------------
# Entropy
# -----------------------------
entropy = -(p_yes * math.log2(p_yes) + p_no * math.log2(p_no))

print("Entropy (Parent):", round(entropy, 3))

# -----------------------------
# Split on Gender (Example)
# Male: Yes=2, No=3
# Female: Yes=5, No=0
# -----------------------------

# Male node
male_yes = 2
male_no = 3
male_total = male_yes + male_no

p_my = male_yes / male_total
p_mn = male_no / male_total

entropy_male = -(p_my * math.log2(p_my) + p_mn * math.log2(p_mn))

# Female node (pure)
entropy_female = 0

# -----------------------------
# Weighted entropy after split
# -----------------------------
weighted_entropy = (
    (male_total / total) * entropy_male +
    (5 / total) * entropy_female
)

# -----------------------------
# Information Gain
# -----------------------------
information_gain = entropy - weighted_entropy

print("Information Gain (Gender):", round(information_gain, 3))


# In[7]:


import math

# -----------------------------
# Calculate Gini Impurity
# -----------------------------
def calculate_gini(groups, classes):
    total = sum(len(group) for group in groups)
    gini = 0.0

    for group in groups:
        size = len(group)
        if size == 0:
            continue

        score = 0.0
        for c in classes:
            p = [row[-1] for row in group].count(c) / size
            score += p * p

        gini += (1 - score) * (size / total)

    return gini


# -----------------------------
# Calculate Entropy
# -----------------------------
def calculate_entropy(groups, classes):
    total = sum(len(group) for group in groups)
    entropy = 0.0

    for group in groups:
        size = len(group)
        if size == 0:
            continue

        group_entropy = 0.0
        for c in classes:
            p = [row[-1] for row in group].count(c) / size
            if p > 0:
                group_entropy -= p * math.log2(p)

        entropy += group_entropy * (size / total)

    return entropy


# -----------------------------
# Split Dataset
# -----------------------------
def split_data(index, threshold, dataset):
    left, right = [], []

    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)

    return left, right


# -----------------------------
# Find Best Split
# -----------------------------
def best_split(dataset):
    classes = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, float("inf"), None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_data(index, row[index], dataset)
            gini = calculate_gini(groups, classes)

            if gini < best_score:
                best_index = index
                best_value = row[index]
                best_score = gini
                best_groups = groups

    return best_index, best_value, best_score, best_groups


# -----------------------------
# Example Dataset
# -----------------------------
dataset = [
    [22, 1],
    [25, 1],
    [30, 1],
    [35, 0],
    [40, 0],
    [45, 0]
]

index, threshold, gini, groups = best_split(dataset)

print("Best Feature Index:", index)
print("Best Threshold:", threshold)
print("Best Gini:", round(gini, 3))


# In[8]:


import random
import math
from collections import Counter

# -----------------------------
# Dataset: Income Prediction
# -----------------------------
dataset = [
    [22, "Bachelors", 40, "LOW"],
    [35, "Masters", 50, "HIGH"],
    [29, "Diploma", 60, "HIGH"],
    [45, "Masters", 45, "HIGH"],
    [25, "Bachelors", 35, "LOW"],
    [32, "Diploma", 40, "LOW"],
    [41, "Masters", 55, "HIGH"],
    [28, "Bachelors", 50, "HIGH"]
]

# Encode categorical values
edu_map = {"Diploma": 0, "Bachelors": 1, "Masters": 2}
label_map = {"LOW": 0, "HIGH": 1}

data = [[r[0], edu_map[r[1]], r[2], label_map[r[3]]] for r in dataset]


# -----------------------------
# Gini Impurity
# -----------------------------
def gini(groups, classes):
    total = sum(len(g) for g in groups)
    g = 0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0
        for c in classes:
            p = [row[-1] for row in group].count(c) / size
            score += p * p
        g += (1 - score) * (size / total)
    return g


# -----------------------------
# Split Dataset
# -----------------------------
def split_data(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# -----------------------------
# Build Decision Tree (Stump)
# -----------------------------
def build_tree(dataset):
    classes = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score = None, None, float("inf")

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_data(index, row[index], dataset)
            score = gini(groups, classes)
            if score < best_score:
                best_index, best_value, best_score = index, row[index], score

    return best_index, best_value


# -----------------------------
# Predict using Decision Tree
# -----------------------------
def predict_tree(tree, row):
    index, value = tree
    return 0 if row[index] < value else 1


# -----------------------------
# Bootstrap Sampling
# -----------------------------
def sample_dataset(dataset):
    return [random.choice(dataset) for _ in range(len(dataset))]


# -----------------------------
# Build Random Forest
# -----------------------------
def build_forest(dataset, n_trees=5):
    forest = []
    for _ in range(n_trees):
        sample = sample_dataset(dataset)
        tree = build_tree(sample)
        forest.append(tree)
    return forest


# -----------------------------
# Predict using Random Forest
# -----------------------------
def predict_forest(forest, row):
    predictions = [predict_tree(tree, row) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]


# -----------------------------
# Evaluation
# -----------------------------
y_true = [row[-1] for row in data]

# Decision Tree
tree = build_tree(data)
dt_preds = [predict_tree(tree, row) for row in data]

# Random Forest
forest = build_forest(data, n_trees=7)
rf_preds = [predict_forest(forest, row) for row in data]

dt_accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == dt_preds[i]) / len(y_true)
rf_accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == rf_preds[i]) / len(y_true)

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = {
    "Blood_Pressure": [120, 140, 130, 160, 110, 150, 145, 125],
    "Sugar_Level": [90, 180, 150, 200, 85, 190, 170, 95],
    "Heart_Rate": [72, 88, 80, 95, 70, 92, 85, 75],
    "Age": [25, 60, 45, 70, 22, 65, 55, 30],
    "Risk": ["Low", "High", "High", "High", "Low", "High", "High", "Low"]
}

df = pd.DataFrame(data)

# Encode target
df["Risk"] = df["Risk"].map({"Low": 0, "High": 1})

X = df.drop("Risk", axis=1)
y = df["Risk"]

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 3: Train Decision Tree
# (with tuning to avoid overfitting)
# -----------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# Step 5: Feature Importance
# -----------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# -----------------------------
# Step 6: Visualize Tree
# -----------------------------
plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Low Risk", "High Risk"],
    filled=True
)
plt.show()


# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Training Dataset
# -----------------------------
data = {
    "Blood_Pressure": [120, 140, 130, 160, 110, 150, 145, 125],
    "Sugar_Level": [90, 180, 150, 200, 85, 190, 170, 95],
    "Heart_Rate": [72, 88, 80, 95, 70, 92, 85, 75],
    "Age": [25, 60, 45, 70, 22, 65, 55, 30],
    "Risk": [0, 1, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("Risk", axis=1)
y = df["Risk"]

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# -----------------------------
# USER INPUT
# -----------------------------
print("\n🏥 Enter Patient Details")

bp = int(input("Enter Blood Pressure: "))
sugar = int(input("Enter Sugar Level: "))
heart = int(input("Enter Heart Rate: "))
age = int(input("Enter Age: "))

# Create DataFrame with SAME column names
user_df = pd.DataFrame([[bp, sugar, heart, age]], columns=X.columns)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(user_df)

if prediction[0] == 1:
    print("\n⚠️ Prediction: HIGH RISK")
else:
    print("\n✅ Prediction: LOW RISK")


# In[14]:


print("🎬 Movie Recommendation Tree 🎬\n")

# User inputs
duration = int(input("Enter movie duration (minutes): "))
actor_popularity = input("Actor popularity (high/low): ").lower()
year = int(input("Year of release: "))
budget = input("Budget (high/low): ").lower()

# Manual Decision Tree Logic
if duration > 120:
    if budget == "high":
        genre = "Action"
    else:
        genre = "Drama"
else:
    if actor_popularity == "high":
        genre = "Comedy"
    else:
        if year > 2015:
            genre = "Comedy"
        else:
            genre = "Drama"

print("\n🎥 Recommended Movie Genre:", genre)


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = {
    "Word_Frequency": [5, 20, 3, 25, 2, 30, 4, 18],
    "Link_Count": [0, 5, 0, 6, 0, 7, 1, 4],
    "Email_Length": [200, 800, 180, 900, 150, 1000, 220, 750],
    "Spam": [0, 1, 0, 1, 0, 1, 0, 1]   # 0 = Not Spam, 1 = Spam
}

df = pd.DataFrame(data)

X = df.drop("Spam", axis=1)
y = df["Spam"]

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 3: Train Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# -----------------------------
# Step 4: Train Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------------
# Step 5: Feature Importance
# -----------------------------
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# -----------------------------
# Step 6: Results
# -----------------------------
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nFeature Importance (Random Forest):")
print(importance)

# -----------------------------
# Step 7: USER INPUT PREDICTION
# -----------------------------
print("\n📧 Enter New Email Details")

word_freq = int(input("Enter Word Frequency: "))
links = int(input("Enter Link Count: "))
length = int(input("Enter Email Length: "))

new_email = pd.DataFrame(
    [[word_freq, links, length]],
    columns=X.columns
)

prediction = rf.predict(new_email)

if prediction[0] == 1:
    print("\n⚠️ Prediction: SPAM EMAIL")
else:
    print("\n✅ Prediction: NOT SPAM EMAIL")


# In[18]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------
# Step 1: Manually Created Dataset
# --------------------------------
# Features:
# Visits, Avg_Spend, Rating, Discount_Used
# Targets:
# Reorder (Yes/No), Menu_Group (Veg/Non-Veg/Dessert)

data = {
    "Visits": [1, 5, 3, 8, 2, 6, 4, 7],
    "Avg_Spend": [200, 500, 300, 700, 250, 600, 400, 650],
    "Rating": [3, 5, 4, 5, 3, 4, 4, 5],
    "Discount_Used": [0, 1, 0, 1, 0, 1, 0, 1],

    "Reorder": [0, 1, 0, 1, 0, 1, 1, 1],          # 0 = No, 1 = Yes
    "Menu_Group": [0, 1, 0, 1, 0, 2, 2, 1]        # 0 = Veg, 1 = Non-Veg, 2 = Dessert
}

df = pd.DataFrame(data)

# --------------------------------
# Step 2: Split Inputs & Outputs
# --------------------------------
X = df[["Visits", "Avg_Spend", "Rating", "Discount_Used"]]

y_reorder = df["Reorder"]
y_menu = df["Menu_Group"]

# --------------------------------
# Step 3: Train-Test Split
# --------------------------------
X_train, X_test, y_train_r, y_test_r = train_test_split(
    X, y_reorder, test_size=0.25, random_state=42
)

_, _, y_train_m, y_test_m = train_test_split(
    X, y_menu, test_size=0.25, random_state=42
)

# --------------------------------
# Step 4: Train Decision Trees
# --------------------------------
reorder_model = DecisionTreeClassifier(max_depth=3, random_state=42)
menu_model = DecisionTreeClassifier(max_depth=3, random_state=42)

reorder_model.fit(X_train, y_train_r)
menu_model.fit(X_train, y_train_m)

# --------------------------------
# Step 5: Accuracy
# --------------------------------
reorder_pred = reorder_model.predict(X_test)
menu_pred = menu_model.predict(X_test)

print("Reorder Prediction Accuracy:",
      accuracy_score(y_test_r, reorder_pred))

print("Menu Group Prediction Accuracy:",
      accuracy_score(y_test_m, menu_pred))

# --------------------------------
# Step 6: USER INPUT (Restaurant App)
# --------------------------------
print("\n🍽️ Enter Customer Details")

visits = int(input("Number of visits: "))
spend = int(input("Average spend: "))
rating = int(input("Customer rating (1–5): "))
discount = int(input("Discount used? (1=Yes, 0=No): "))

user_data = pd.DataFrame(
    [[visits, spend, rating, discount]],
    columns=X.columns
)

# Predictions
reorder_result = reorder_model.predict(user_data)[0]
menu_result = menu_model.predict(user_data)[0]

# Output
print("\n📊 Prediction Results")

print("Will customer reorder?",
      "YES ✅" if reorder_result == 1 else "NO ❌")

menu_map = {0: "Veg 🥗", 1: "Non-Veg 🍗", 2: "Dessert 🍰"}
print("Preferred Menu Group:", menu_map[menu_result])


# In[19]:


print("🎬 What Should I Watch Today? 🎬\n")

# User inputs (features)
mood = input("Enter your mood (happy/sad/bored): ").lower()
time = input("Time available (short/long): ").lower()
genre = input("Genre preference (action/comedy/drama): ").lower()
popularity = input("Do you want something popular? (yes/no): ").lower()

# Manual Decision Tree Logic
if mood == "happy":
    if time == "short":
        recommendation = "Comedy Movie 😂"
    else:
        recommendation = "Popular Movie ⭐"

else:
    if mood == "sad":
        recommendation = "Drama Movie 🎭"
    else:
        if genre == "action":
            recommendation = "Action Series 💥"
        else:
            recommendation = "Light Comedy 😊"

print("\n🎥 Recommendation:", recommendation)


# In[ ]:




