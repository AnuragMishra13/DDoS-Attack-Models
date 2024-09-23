import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
import os


def load_data(file_path):
    """
    Load and preprocess the test data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe.
    """
    df = pd.read_csv(file_path)
    try:
        encoder = load("encoder(netbios).joblib")
        df["Label"] = encoder.transform(df["Label"])
    except FileNotFoundError:
        print("Encoder file not found. Make sure it exists.")
        raise
    return df


def prepare_data(df):
    """
    Prepare features and labels from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe with data.

    Returns:
        X, y: Features and labels as numpy arrays.
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def load_model(model_path):
    """
    Load the model from disk.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        model: Loaded model or None if not found.
    """
    try:
        model = load(model_path)
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Skipping...")
        return None
    return model


def evaluate_model(model, X, y, results_dir, model_name):
    """
    Evaluate the model and generate reports and curves.

    Args:
        model: Trained model.
        X (numpy array): Features for evaluation.
        y (numpy array): True labels.
        results_dir (str): Directory to save the results.
        model_name (str): Name of the model (for saving results).
    """
    # Predict labels and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Assuming binary classification

    # Generate and print the classification report
    report = classification_report(y, y_pred)
    print(report)

    # Save the classification report to a file
    with open(f"{results_dir}/Classification_Report.txt", "a") as file:
        file.write(f"Classification Report for {model_name}\n{report}\n{'=' * 53}\n")

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/{model_name}_roc.png")
    plt.close()

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_proba)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color="b", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(f"{results_dir}/{model_name}_precision_recall_curve.png")
    plt.close()


def main():
    # Create results directory if it doesn't exist
    results_dir = "Results/Test Results"
    os.makedirs(results_dir, exist_ok=True)

    # Load and preprocess the test data
    df = load_data("../Testing_datasets/NetBIOS.csv")

    # Prepare features and labels
    X, y = prepare_data(df)

    # Load and apply the scaler
    scaler = load("../standard_scaler.joblib")
    X = scaler.transform(X)

    # Models dictionary (Add more models as needed)
    models = {"CatBoost": "Models/catboost_model.joblib"}

    for model_name, model_path in models.items():
        # Load the model
        model = load_model(model_path)
        if model is None:
            continue

        # Evaluate the model
        evaluate_model(model, X, y, results_dir, model_name)


if __name__ == "__main__":
    main()
