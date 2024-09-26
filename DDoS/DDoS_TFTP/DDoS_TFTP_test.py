import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from joblib import load
import tempfile

# Directories
result_dir = os.path.join("Results", "Train_Result")
model_dir = os.path.join("Models")
os.makedirs(result_dir, exist_ok=True)

models = {
        "CatBoostClassifier": "Models/CatBoostClassifier_model.joblib"
    }

def evaluate_models(models):
    """
    Evaluate the models and generate reports and plots.

    Args:
        models (dict): Dictionary containing models to be evaluated.
    """
    for model_name,model_path in models.items():
        print(f"Evaluating {model_name}")

        # Load the trained model
        loaded_model = load(model_path)

        X_test = np.load("Combined_Data/X_test.npy")
        y_test = np.load("Combined_Data/y_test.npy")

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

# Main function for evaluation
def main(): # Path to the saved test file
    evaluate_models(models)

if __name__ == "__main__":
    main()
