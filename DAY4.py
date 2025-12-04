#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")
df.head()
plt.figure(figsize=(7,4))
plt.hist(df["sepal_length"], bins=20)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Insight:
# Most flowers have sepal lengths between 5 and 6 cm.
plt.figure(figsize=(7,4))
sns.barplot(x="species", y="petal_length", data=df)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Insight:
# Virginica species has the highest average petal length, followed by Versicolor and Setosa.
plt.figure(figsize=(7,4))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Insight:
# Virginica shows a clear positive relationship — longer sepals come with longer petals.
plt.figure(figsize=(7,4))
plt.plot(df["petal_width"][:30], marker="o")
plt.title("Trend of Petal Width for First 30 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Width (cm)")
plt.show()

# Insight:
# Petal width increases sharply after the first 10 samples, indicating species difference.
plt.figure(figsize=(7,4))
sns.boxplot(x="species", y="sepal_width", data=df)
plt.title("Comparison of Sepal Width Across Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Insight:
# Setosa has the highest median sepal width with fewer outliers compared to other species.
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()

# Insight:
# Petal length and petal width show very strong correlation (~0.96).


# In[7]:


import pandas as pd
df = pd.read_csv("StudentsPerformance.csv")
df.head()


# In[8]:


import pandas as pd
df = pd.read_csv("StudentsPerformance.csv")
df.head()


# In[9]:


import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")      # ← change file name here

# Display first 5 rows
print("📌 First 5 rows:")
display(df.head())

# Dataset shape
print("\n📌 Shape (Rows, Columns):")
print(df.shape)

# Column names
print("\n📌 Column Names:")
print(df.columns.tolist())

# Dataset info (data types + null overview)
print("\n📌 Dataset Info:")
df.info()

# Basic Statistical Summary (numerical columns)
print("\n📌 Descriptive Statistics:")
display(df.describe())

# Count Missing Values per Column
print("\n📌 Missing Values per Column:")
print(df.isnull().sum())

# Count Unique Values per Column
print("\n📌 Unique Values per Column:")
print(df.nunique())


# In[ ]:




