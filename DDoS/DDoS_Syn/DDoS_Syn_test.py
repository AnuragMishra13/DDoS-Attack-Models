#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import os
import gc

# Create results directory if it doesn't exist
results_dir = 'Results'
os.makedirs(results_dir, exist_ok=True)

# Load and preprocess the test data
df = pd.read_csv("Testing_datasets/Syn.csv")
df = df.drop(columns=['Unnamed: 0', ' Timestamp', 'SimillarHTTP', 'Flow ID', ' Source IP', ' Destination IP'])
df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Prepare features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Load and apply the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

X = scaler.transform(X)

# Load the CatBoost model
model = load('catboost_model.pkl')

# Predict labels and probabilities
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]  # Assuming binary classification, use probability of positive class

# Print classification report
print(classification_report(y, y_pred))

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig(f'{results_dir}/roc.png')
plt.close()

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y, y_proba)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(f'{results_dir}/precision_recall_curve.png')
plt.close()

# Cleanup
del model
gc.collect()