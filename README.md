# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
 # -*- coding: utf-8 -*-
"""EX NO6:Data visualization Using Seaborn"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------------
# **LINE PLOT**
# --------------------------------------------------------

x = [1, 2, 3, 4, 5]
y1 = [3, 5, 2, 6, 1]
y2 = [1, 6, 4, 3, 8]
y3 = [5, 2, 7, 1, 4]

plt.figure(figsize=(8,5))
sns.lineplot(x=x, y=y1, marker='o', label='Line 1')
sns.lineplot(x=x, y=y2, marker='s', label='Line 2')
sns.lineplot(x=x, y=y3, marker='^', label='Line 3')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')
plt.legend()
plt.show()


# --------------------------------------------------------
# **BAR PLOT - Seaborn**
# --------------------------------------------------------

# Using tips dataset
tips = sns.load_dataset('tips')

# Basic bar plot with hue
plt.figure(figsize=(8,5))
sns.barplot(x='day', y='total_bill', data=tips, hue='sex', palette='pastel')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')
plt.show()


# --------------------------------------------------------
# **TITANIC DATASET BAR PLOT**
# --------------------------------------------------------

# Load Titanic dataset
tit = pd.read_csv("/content/titanic_dataset.csv")

# 1. Fare by Embarked
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked', y='Fare', data=tit, palette='rainbow')
plt.title("Fare of Passenger by Embarked Town")
plt.show()

# 2. Fare by Embarked, Divided by Class
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked', y='Fare', data=tit, palette='rainbow', hue='Pclass')
plt.title("Fare of Passenger by Embarked Town, Divided by Class")
plt.show()


# --------------------------------------------------------
# **SCATTER PLOT**
# --------------------------------------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', style='time', s=100)
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
plt.show()


# --------------------------------------------------------
# **VIOLIN PLOT**
# --------------------------------------------------------

plt.figure(figsize=(8,5))
sns.violinplot(x='Pclass', y='Fare', data=tit, palette='Set2')
plt.title('Violin Plot of Fare by Passenger Class')
plt.show()


# --------------------------------------------------------
# **HISTOGRAM**
# --------------------------------------------------------

# Random marks dataset
np.random.seed(0)
marks = np.random.normal(loc=70, scale=10, size=100)

plt.figure(figsize=(8,5))
sns.histplot(marks, kde=True, color='skyblue', bins=15)
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.title('Histogram of Marks')
plt.show()


# Histogram for Titanic dataset: Pclass vs Survived
plt.figure(figsize=(8,5))
sns.histplot(data=tit, x='Pclass', hue='Survived', multiple='stack', kde=True, palette='Set1')
plt.title('Histogram of Pclass Divided by Survived')
plt.show()


# Result:
 
![alt text](<Screenshot 2025-10-08 162615.png>)