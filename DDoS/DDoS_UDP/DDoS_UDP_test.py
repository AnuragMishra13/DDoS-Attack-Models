import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from joblib import load
import gc
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
import os
import matplotlib.pyplot as plt

# Create results directory if it doesn't exist
results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)


# Function to process and predict a dataset chunk
def process_chunk(chunk, label_encoder, scaler, model):
    # Drop unnecessary columns and handle missing values
    chunk = chunk.drop(
        columns=[
            "Unnamed: 0",
            " Timestamp",
            "SimillarHTTP",
            "Flow ID",
            " Source IP",
            " Destination IP",
        ]
    )
    chunk.columns = chunk.columns.str.strip()
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk = chunk.dropna()

    # Encode labels
    chunk["Label"] = label_encoder.transform(chunk["Label"])

    # Prepare features and labels
    X = chunk.iloc[:, :-1].values
    y = chunk.iloc[:, -1].values

    # Scale features
    X = scaler.transform(X)

    # Predict labels and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[
        :, 1
    ]  # Assuming binary classification, use probability of positive class

    return y, y_pred, y_proba


# Load the scaler and models
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

model_paths = [
    f"Models/{model_name}.joblib"
    for model_name in [
        "CatBoost",
        "RandomForest",
        "XGBoost",
        "GradientBoosting",
        "LightGBM",
        "AdaBoost",
    ]
]

models = {
    model_name: load(model_path)
    for model_name, model_path in zip(
        [
            "CatBoost",
            "RandomForest",
            "XGBoost",
            "GradientBoosting",
            "LightGBM",
            "AdaBoost",
        ],
        model_paths,
    )
}

# Process the dataset in chunks
chunk_size = 500000
chunks = pd.read_csv("../Testing_datasets/UDP.csv", chunksize=chunk_size)
half_size = 0

for model_name, model in models.items():
    predictions = []

    # Reset label encoder for each model
    label_encoder = LabelEncoder()

    # Process the dataset in two halves
    for i, chunk in enumerate(chunks):
        if i == half_size:
            # Free memory and process the second half
            del chunk
            gc.collect()
            half_size = i  # Update the half size for the second pass
            continue

        # Fit label encoder on the chunk
        label_encoder.fit(
            chunk[" Label"]
        )  # Fit on the chunk to ensure consistent encoding

        y, y_pred, y_proba = process_chunk(chunk, label_encoder, scaler, model)
        predictions.append((y, y_pred, y_proba))

        # Free memory
        del chunk
        gc.collect()

    # Check if predictions are empty
    if not predictions:
        print(f"No predictions were made for {model_name}.")
        continue

    # Combine predictions from all chunks
    try:
        y_true, y_pred, y_proba = zip(*predictions)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_proba = np.concatenate(y_proba)
    except ValueError as e:
        print(f"Error combining predictions for {model_name}: {e}")
        continue

    # Print classification report
    print(f"Results for {model_name}:")
    print(classification_report(y_true, y_pred))

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/roc_{model_name}.png")
    plt.close()

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color="b", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(f"{results_dir}/precision_recall_curve_{model_name}.png")
    plt.close()

    # Cleanup
    del model
    gc.collect()
