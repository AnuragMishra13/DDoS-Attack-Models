#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load and preprocess data
df = pd.read_csv("Traning Datasets/Syn.csv")
df = df.drop(columns=['Unnamed: 0', ' Timestamp', 'SimillarHTTP', 'Flow ID', ' Source IP', ' Destination IP'])
df.columns = df.columns.str.strip()

# Display the shape of the dataframe
print(df.shape)

# Display class counts
class_counts = df['Label'].value_counts()
print(class_counts)

# Handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Display updated shape and class counts
print(df.shape)
class_counts = df['Label'].value_counts()
print(class_counts)

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Prepare features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=27)
X, y = smote.fit_resample(X, y)

# Scale features
scaler = MaxAbsScaler()
X = scaler.fit_transform(X)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save the model
dump(model, "Models/Syn_model.joblib")

# Display final class counts
class_counts = df['Label'].value_counts()
print(class_counts)
