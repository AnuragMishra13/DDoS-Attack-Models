# Import necessary libraries
import numpy as np
import pandas as pd
import gc
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Set environment variables for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device ID if needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Define the results directory
results_dir = "Results/Train Results"
os.makedirs(results_dir, exist_ok=True)

# Load pre-trained scaler and encoder
scaler = load("../standard_scaler.joblib")
encoder = LabelEncoder()

# Load the dataset
df = pd.read_csv("../Training_Datasets/MSSQL.csv")

# Encode labels
df["Label"] = encoder.fit_transform(df["Label"])
dump(encoder, "encoder(mssql).joblib")

# Prepare features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Clean up dataframe to free memory
del df
gc.collect()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.33
)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=27)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Transform the data using the scaler
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Clean up to free memory
del X, y
gc.collect()


# Initialize classifiers with parallel processing and GPU support where applicable
def initialize_classifiers():
    """
    Initialize classifiers with parallel processing and GPU support where applicable.

    Returns:
        dict: A dictionary containing the initialized classifiers.
    """
    models = {
        "catboost": CatBoostClassifier(
            task_type="GPU", devices="0", thread_count=-1, gpu_ram_part=0.95
        ),  # Set to 95% of your VRAM
    }
    return models


# Train and save models
def train_and_save_models(models):
    """
    Train and save the models.

    Args:
        models (dict): A dictionary containing the models to be trained.
    """
    for model_name, model in models.items():
        print(f"Training {model_name}")

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Save the trained model to disk
        model_path = f"Models/{model_name}_model.joblib"
        dump(model, model_path)
        print(f"{model_name} model trained and saved as {model_path}")

        # Clean up to free memory
        del model
        gc.collect()


# Evaluate the models
def evaluate_models(models):
    """
    Evaluate the models.

    Args:
        models (dict): A dictionary containing the models to be evaluated.
    """
    for model_name in models.keys():
        print(f"Results for {model_name} : \n")

        # Load the model from file
        model_path = f"Models/{model_name}_model.joblib"
        loaded_model = load(model_path)

        # Make predictions on the test set
        y_pred = loaded_model.predict(X_test)
        y_proba = loaded_model.predict_proba(X_test)[
            :, 1
        ]  # Probability of positive class

        # Generate and print the classification report
        report = classification_report(y_test, y_pred)
        print(report)

        # Save the classification report to a file
        with open(f"{results_dir}/Classification_Report.txt", "a") as file:
            file.write(
                f"Classification Report for {model_name}\n{report}\n{'=' * 53}\n"
            )

        # Compute ROC curve and AUC score
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Plot and save ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
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
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        # Plot and save Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(f"{results_dir}/{model_name}_precision_recall_curve.png")
        plt.close()


# Main function
def main():
    models = initialize_classifiers()
    train_and_save_models(models)
    evaluate_models(models)


if __name__ == "__main__":
    main()
