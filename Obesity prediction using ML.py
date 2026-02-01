# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:36:41 2025

@author: Ajeet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


#Load the dataset

df = pd.read_csv(r"G:\Predictive_analytics\ObesityDataSet_raw_and_data_sinthetic.csv")

#See the structure of our dataset

print(df.head())

print(df.describe())

print(df.info())

# Quick initial checks

print("Shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

#Drop exact duplicate rows

df = df.drop_duplicates().reset_index(drop=True)

# Convert Age and Height to numeric first

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Height'] = pd.to_numeric(df['Height'], errors='coerce')

# Age should be an integer

df['Age'] = df['Age'].round(0).astype('Int64')

# Height rounded to 2 decimals

df['Height'] = df['Height'].round(2)

#Convert to numeric column

cols_to_int = ['Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for col in cols_to_int:
    if col in df.columns:
        
        df[col] = pd.to_numeric(df[col], errors='coerce')

        df[col] = df[col].round(0)

        df[col] = df[col].astype('Int64')


#Remove leading/trailing spaces from all string columns

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

#Convert all string columns to lowercase and strip spaces

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()
    
#Replace underscores with spaces (for readability)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.replace('_', ' ')

# check how many NaNs after coercion

print(df[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']].isna().sum())

#Convert categorical object columns to category dtype (saves memory and helps later encoding)

cat_cols = df.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    df[c] = df[c].astype('category')

#Create BMI (if not present) — obesity datasets often benefit from BMI.

# If Height currently in meters:
df['BMI'] = df['Weight'] / (df['Height']**2)
# If Height in cm:
# df['BMI'] = df['Weight'] / ((df['Height']/100)**2)

df['BMI'] = df['BMI'].round(2)
print(df['BMI'].describe())

#Value counts for each categorical column

for col in df.select_dtypes(include='object').columns:
    print(col, ":\n", df[col].value_counts(), "\n")
    
#Outlier check for numeric features,

def iqr_bounds(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5*iqr, q3 + 1.5*iqr

for col in ['BMI','Weight','Height','Age']:
    if col in df.columns:
        low, high = iqr_bounds(df[col].dropna())
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        print(col, "outliers:", n_out, "bounds:", low, high)

#Quick overview (shape, dtypes, missing, unique categories, re-check duplicates)

print("Shape:", df.shape)
print(df.dtypes)
print("Missing per column:\n", df.isnull().sum())
print("Duplicates after cleaning:", df.duplicated().sum())
print(df.head())
print(df.describe())
print(df.info())
print("\nUnique values in categorical columns:")
for c in cat_cols:
    print(c, df[c].nunique())
    
#Target identification & distribution

target_col = 'NObeyesdad'
print(df[target_col].value_counts())
df[target_col].value_counts().plot(kind='bar', title='Target distribution')
plt.show()

    
#Univariate plots for numeric features (histograms + boxplots)

num_cols = df.select_dtypes(include=['int64','float64','Int64']).columns.tolist()

# histograms

df[num_cols].hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# boxplot for BMI specifically

if 'BMI' in df.columns:
    plt.figure()
    df.boxplot(column='BMI')
    plt.title('BMI boxplot')
    plt.show()
    
#Categorical feature counts (bar plots)

for c in cat_cols:
    plt.figure()
    df[c].value_counts().plot(kind='bar')
    plt.title(f"{c} counts")
    plt.show()

#Bivariate analysis: numeric vs target

# violin/boxplot of BMI by target

if 'BMI' in df.columns and target_col in df.columns:
    plt.figure()
    sns.boxplot(x=target_col, y='BMI', data=df)
    plt.title('BMI by target')
    plt.show()

#Pairwise relationships for a small subset (scatter + correlation)

# choose a few numeric features to plot pairwise

cols = ['BMI','Age','Height','Weight','CH2O']  
cols = [c for c in cols if c in df.columns]
sns.pairplot(df[cols].sample(500))   
plt.show()

# correlation heatmap

plt.figure(figsize=(8,6))
corr = df[cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

#Categorical vs target (crosstab / stacked bar)

# Example for Gender vs target

if 'Gender' in df.columns:
    ct = pd.crosstab(df['Gender'], df[target_col], normalize='index')  
    print(ct)
    ct.plot(kind='bar', stacked=True)
    plt.title('Gender vs Target (proportion)')
    plt.show()
 
#BMI vs Obesity Level (Mean BMI per Class)

df.groupby('NObeyesdad')['BMI'].mean().plot(kind='bar')
plt.title("Average BMI Across Obesity Categories")
plt.ylabel("Mean BMI")
plt.xlabel("Obesity Level")
plt.show()

#Age Distribution by Obesity Level

plt.figure(figsize=(8,5))
sns.boxplot(x='NObeyesdad', y='Age', data=df)
plt.title("Age Distribution Across Obesity Levels")
plt.xticks(rotation=45)
plt.show()

#Prepare X and Y

target_col = 'NObeyesdad'
y = df[target_col].astype('category')
X = df.drop(columns=[target_col])

# Quick check
print("X shape:", X.shape, "y shape:", y.shape)
print("Target classes:", y.unique())

#Train - test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Train:", X_train.shape, "Test:", X_test.shape)


#Encode features (same as before)

# Encode target
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# One-hot encode categorical features
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align train and test columns
X_train_enc, X_test_enc = X_train_enc.align(
    X_test_enc, join='left', axis=1, fill_value=0
)

#Normalization

scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train_enc)
X_test_norm = scaler.transform(X_test_enc)

print("Normalization applied (0–1 scale)")

#Train ML models

#KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_norm, y_train_enc)
y_pred_knn = knn.predict(X_test_norm)

#Naive Bayes

nb = GaussianNB()
nb.fit(X_train_norm, y_train_enc)
y_pred_nb = nb.predict(X_test_norm)


#Decision Tree Classifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_enc, y_train_enc)   
y_pred_dt = dt.predict(X_test_enc)


#SVM

svm = SVC(kernel='rbf')
svm.fit(X_train_norm, y_train_enc)
y_pred_svm = svm.predict(X_test_norm)

#Compare Model Performance 


results = []

def evaluate(name, y_true, y_pred):
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro'),
        "Recall": recall_score(y_true, y_pred, average='macro'),
        "F1 Score": f1_score(y_true, y_pred, average='macro')
    })

evaluate("KNN", y_test_enc, y_pred_knn)
evaluate("Naive Bayes", y_test_enc, y_pred_nb)
evaluate("Decision Tree", y_test_enc, y_pred_dt)
evaluate("SVM", y_test_enc, y_pred_svm)

results_df = pd.DataFrame(results)
print(results_df)

#KNN Confusion Matrix

cm_knn = confusion_matrix(y_test_enc, y_pred_knn)

plt.figure(figsize=(6,5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Naive Bayes Confusion Matrix

cm_nb = confusion_matrix(y_test_enc, y_pred_nb)

plt.figure(figsize=(6,5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Decision tree Classifier

cm_dt = confusion_matrix(y_test_enc, y_pred_dt)

plt.figure(figsize=(6,5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#SVM confusion Matrix

cm_svm = confusion_matrix(y_test_enc, y_pred_svm)

plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Confusion Matrices for All Models (Side-by-Side)

models = {
    "KNN": y_pred_knn,
    "Naive Bayes": y_pred_nb,
    "Decision Tree": y_pred_dt,
    "SVM": y_pred_svm
}

plt.figure(figsize=(12,8))

for i, (name, preds) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test_enc, preds)
    
    plt.subplot(2,2,i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.suptitle("Confusion Matrices of All Models", fontsize=14)
plt.tight_layout()
plt.show()


#Accuracy comparison Bar Graph

results_df.set_index("Model")["Accuracy"].plot(
    kind='bar',
    figsize=(7,5),
    color=['blue','green','orange','purple']
)

plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Models")
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()

#Prediction Distribution of Best Model (SVM)

pd.Series(y_pred_svm).value_counts().plot(kind='bar')
plt.title("Predicted Obesity Class Distribution (SVM)")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()

# Predicted Obesity Class Distribution - Decision Tree

# Convert predictions to Series for easy plotting
dt_pred_series = pd.Series(y_pred_dt)

# Plot
plt.figure(figsize=(7,5))
dt_pred_series.value_counts().sort_index().plot(kind='bar')

plt.title("Predicted Obesity Class Distribution Using Decision Tree")
plt.xlabel("Predicted Obesity Class")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()


#Confusion Matrices for All Models (Side-by-Side)

models = {
    "KNN": y_pred_knn,
    "Naive Bayes": y_pred_nb,
    "Decision Tree": y_pred_dt,
    "SVM": y_pred_svm
}

plt.figure(figsize=(12,8))

for i, (name, preds) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test_enc, preds)
    
    plt.subplot(2,2,i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.suptitle("Confusion Matrices of All Models", fontsize=14)
plt.tight_layout()
plt.show()


# Performance comparison bar chart
metrics_df = results_df.set_index("Model")

metrics_df[['Accuracy','Precision','Recall','F1 Score']].plot(
    kind='bar',
    figsize=(8,5)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()

#DASHBOARD 

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Create Main Window
root = tk.Tk()
root.title("Obesity Prediction – ML Dashboard")
root.geometry("1000x750")

#Title
tk.Label(
    root,
    text="Obesity Prediction Project Dashboard",
    font=("Arial", 20, "bold")
).pack(pady=10)

#Best Model Highlight
best = results_df.loc[results_df["Accuracy"].idxmax()]

tk.Label(
    root,
    text=f"🏆 Best Model: {best['Model']}  |  Accuracy: {best['Accuracy']:.4f}",
    font=("Arial", 14),
    fg="green"
).pack(pady=5)

#Model Performance Table
frame_table = tk.Frame(root)
frame_table.pack(pady=10)

tk.Label(
    frame_table,
    text="Model Performance Metrics",
    font=("Arial", 14, "bold")
).pack()

cols = list(results_df.columns)
tree = ttk.Treeview(frame_table, columns=cols, show="headings")

for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=120)

for _, row in results_df.iterrows():
    tree.insert("", tk.END, values=list(row))

tree.pack()

#Accuracy Comparison Graph 
fig_acc, ax_acc = plt.subplots(figsize=(6,4))
ax_acc.bar(results_df["Model"], results_df["Accuracy"], color="skyblue")
ax_acc.set_ylim(0,1)
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Model Accuracy Comparison")

canvas_acc = FigureCanvasTkAgg(fig_acc, master=root)
canvas_acc.draw()
canvas_acc.get_tk_widget().pack(pady=10)

#Confusion Matrix Section 
frame_cm = tk.Frame(root)
frame_cm.pack(pady=10)

tk.Label(
    frame_cm,
    text="Select Model for Confusion Matrix:",
    font=("Arial", 12)
).pack(side=tk.LEFT, padx=5)

combo = ttk.Combobox(
    frame_cm,
    values=["KNN", "Naive Bayes", "Decision Tree", "SVM"],
    state="readonly"
)
combo.current(0)
combo.pack(side=tk.LEFT, padx=5)

cm_canvas = None  # to avoid overlapping plots

def show_confusion():
    global cm_canvas

    if cm_canvas:
        cm_canvas.get_tk_widget().destroy()

    model = combo.get()

    preds = {
        "KNN": y_pred_knn,
        "Naive Bayes": y_pred_nb,
        "Decision Tree": y_pred_dt,
        "SVM": y_pred_svm
    }

    cm = confusion_matrix(y_test_enc, preds[model])

    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix – {model}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    cm_canvas = FigureCanvasTkAgg(fig_cm, master=root)
    cm_canvas.draw()
    cm_canvas.get_tk_widget().pack(pady=10)

tk.Button(
    frame_cm,
    text="Show Confusion Matrix",
    command=show_confusion,
    bg="blue",
    fg="white",
    font=("Arial", 11)
).pack(side=tk.LEFT, padx=5)

#Run Dashboard 
root.mainloop()




