#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from joblib import dump
import gc
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix

# Function to optimize memory usage by changing data types
def optimize_memory(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df

# Create a directory to save models
os.makedirs("Models", exist_ok=True)

# Process data in chunks
chunk_size = 500000
chunks = pd.read_csv("../DrDoS_UDP.csv", chunksize=chunk_size)

processed_chunks = []
minority_class_records = []
label_encoder = LabelEncoder()  # Initialize LabelEncoder

for chunk in chunks:
    # Drop unnecessary columns and handle missing values
    chunk = chunk.drop(columns=["Unnamed: 0", " Timestamp", "SimillarHTTP", "Flow ID", " Source IP", " Destination IP"])
    chunk.columns = chunk.columns.str.strip()
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk = chunk.dropna()

    # Optimize memory usage
    chunk = optimize_memory(chunk)

    # Collect records of the minority class
    minority_class = chunk["Label"].value_counts().idxmin()
    minority_records = chunk[chunk["Label"] == minority_class]
    minority_class_records.append(minority_records)

    # Append processed chunk to list
    processed_chunks.append(chunk)

# Concatenate all processed chunks into a single DataFrame
df_all = pd.concat(processed_chunks, ignore_index=True)

# Fit label encoder on all labels
label_encoder.fit(df_all["Label"])

# Encode labels
df_all["Label"] = label_encoder.transform(df_all["Label"])

# Combine minority class records with a random sample of the majority class
minority_records = pd.concat(minority_class_records, ignore_index=True)
minority_records["Label"] = label_encoder.transform(minority_records["Label"])  # Ensure minority records are encoded

majority_records = df_all[df_all["Label"] != minority_class]
majority_sample = majority_records.sample(n=len(minority_records) * 1, random_state=27)  # Adjust the multiplier as needed

# Combine and shuffle the final dataset
train_df = pd.concat([minority_records, majority_sample], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=27).reset_index(drop=True)  # Shuffle

# Prepare features and labels
X = train_df.drop(columns=["Label"]).values
y = train_df["Label"].values

# Free memory
del train_df
gc.collect()

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=27)
X_smote, y_train = smote.fit_resample(X, y)

# Convert to sparse matrix to save memory
X_sparse = csr_matrix(X_smote)

# Free memory
del X, y
gc.collect()

# Scale features
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_sparse)

# Save scaler and label encoder
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

models = {
    "CatBoost": CatBoostClassifier(learning_rate=0.1, depth=6, iterations=500, random_state=27),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=27),
    "XGBoost": XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=27),
    "GradientBoosting": GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=27),
    "LightGBM": LGBMClassifier(learning_rate=0.1, n_estimators=100, random_state=27),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=27)
}

for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join("Models", f"{model_name}.joblib")
    dump(model, model_path)
    
    # Free memory
    del model
    gc.collect()
