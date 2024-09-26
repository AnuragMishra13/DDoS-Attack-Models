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
    # Load and preprocess the test data
    df = pd.read_csv(file_path)
    encoder = load("encoder(udplag).joblib")
    df["Label"] = encoder.transform(df["Label"])
    return df


def prepare_data(df):
    # Prepare features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def load_model(model_path):
    try:
        # Load the model
        model = load(model_path)
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Skipping...")
        return None
    return model


def evaluate_model(model, X, y, results_dir, model_name):
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
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
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
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{results_dir}/{model_name}_precision_recall_curve.png")
    plt.close()


def main():
    # Create results directory if it doesn't exist
    results_dir = "Results/Test Results"
    os.makedirs(results_dir, exist_ok=True)

    # Load and preprocess the test data
    df = load_data("../Testing_datasets/UDPLag.csv")

    # Prepare features and labels
    X, y = prepare_data(df)

    # Load and apply the scaler
    X = load("../standard_scaler.joblib").transform(X)

    models = {
        "CatBoost": "Models/catboost_model.joblib",
        "ExtraTrees": "Models/extra_trees_model.joblib",
    }

    for model_name, model_path in models.items():
        # Load the model
        model = load_model(model_path)
        if model is None:
            continue

        # Evaluate the model
        evaluate_model(model, X, y, results_dir, model_name)


if __name__ == "__main__":
    main()
