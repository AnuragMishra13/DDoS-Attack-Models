#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import category_encoders as ce

# Load and preprocess data
df = pd.read_csv("Traning Datasets/DrDoS_DNS.csv").drop(columns=['Unnamed: 0', ' Timestamp', 'SimillarHTTP'])
df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Initialize KFold and TargetEncoder
kf = KFold(n_splits=5, shuffle=True, random_state=42)
encoder = ce.TargetEncoder(cols=['Flow ID', 'Source IP', 'Destination IP'])
batch_size = 100000

def process_in_batches(train_df, test_df):
    """
    Process the test data in batches to reduce memory usage.
    """
    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        batch_test_df = test_df.iloc[start:end]
        
        # Transform the batch and update the columns in place
        transformed_batch = encoder.transform(batch_test_df[['Flow ID', 'Source IP', 'Destination IP']])
        df.loc[batch_test_df.index, ['Flow ID', 'Source IP', 'Destination IP']] = transformed_batch.values

# K-Fold processing
for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    # Fit the encoder on the training data
    encoder.fit(train_df[['Flow ID', 'Source IP', 'Destination IP']], train_df['Label'])
    
    # Process the test data in batches and update the columns in place
    process_in_batches(train_df, test_df)

# Save the processed data
df.to_csv("Processed Datasets/DrDoS_DNS.csv", index=False)

# Output class counts and data shape
class_counts = df['Label'].value_counts()
print(class_counts)

print(df.shape)
