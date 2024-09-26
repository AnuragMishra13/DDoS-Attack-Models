import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from catboost import CatBoostClassifier
from joblib import load, dump

# Directories
result_dir = os.path.join("Results", "Train_Result")
model_dir = os.path.join("Models")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("../Training_datasets/SNMP.csv")

# Label encoding
encoder = LabelEncoder()
df["Label"] = encoder.fit_transform(df["Label"])
dump(encoder, "encoder(snmp).joblib")

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Clean up memory
del df
gc.collect()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Further memory cleanup
del X, y
gc.collect()

# Apply SMOTE for class balancing
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Load scaler and scale the data
scaler = load(os.path.join("..", "standard_scaler.joblib"))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Initialize classifiers
def initialize_classifiers():
    """
    Initialize the classifiers with desired configurations.

    Returns:
        dict: Dictionary containing initialized models.
    """
    models = {
        "CatBoostClassifier": CatBoostClassifier(
            task_type="GPU",
            devices="0",
            thread_count=-1,
            gpu_ram_part=0.80,  # Adjust this if needed
            iterations=575,
            learning_rate=0.012068580651907083,
            depth=9,
            l2_leaf_reg=1.5787157652653196,
            bagging_temperature=0.6015896417309138,
            random_strength=2,
        )
    }
    return models


# Train and save models
def train_and_save_models(models):
    """
    Train and save the models.

    Args:
        models (dict): Dictionary of models to be trained.
    """
    for model_name, model in models.items():
        print(f"Training {model_name}")

        # Fit the model
        model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        dump(model, model_path)
        print(f"{model_name} model trained and saved at {model_path}")

        # Free up memory
        del model
        gc.collect()


# Evaluate models
def evaluate_models(models):
    """
    Evaluate the models and generate reports and plots.

    Args:
        models (dict): Dictionary containing models to be evaluated.
    """
    for model_name in models.keys():
        print(f"Evaluating {model_name}")

        # Load the trained model
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        loaded_model = load(model_path)

        # Predict
        y_pred = loaded_model.predict(X_test)
        y_proba = loaded_model.predict_proba(X_test)[
            :, 1
        ]  # Probability of positive class

        # Generate classification report
        report = classification_report(y_test, y_pred)
        print(report)

        # Save classification report
        with open(os.path.join(result_dir, "Classification_Report.txt"), "a") as file:
            file.write(
                f"Classification Report for {model_name}\n{report}\n{'=' * 53}\n"
            )

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(result_dir, f"{model_name}_roc.png"))
        plt.close()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        # Plot Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(
            os.path.join(result_dir, f"{model_name}_precision_recall_curve.png")
        )
        plt.close()


# Main function
def main():
    models = initialize_classifiers()
    train_and_save_models(models)
    evaluate_models(models)


if __name__ == "__main__":
    main()
