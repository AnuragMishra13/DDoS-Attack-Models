import numpy as np
import pandas as pd
import gc
import os
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
import optuna
import matplotlib.pyplot as plt

# Set environment variables for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device ID if needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Prevent full GPU memory allocation

# Define directories for results and models
results_dir = "Results/HyperParameter Results"
models_dir = "Models/HyperParameter Model"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# # Load pre-trained scaler and encoder
# scaler = load("../standard_scaler.joblib")
# encoder = load("encoder(udplag).joblib")

# Load pre-processed train and test data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Ensure no memory is wasted by explicitly deleting unused variables
gc.collect()


# Define the objective function for Optuna hyperparameter tuning
def objective(trial):
    # Define the hyperparameters to tune
    param = {
        "iterations": trial.suggest_int("iterations", 500, 1000),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),  # Updated
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5, log=True),  # Updated
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.2, 1.0, log=True
        ),  # Updated
        "random_strength": trial.suggest_int("random_strength", 1, 3),
        "task_type": "GPU",  # Leverage GPU for faster training
        "devices": "0",  # Specify GPU device ID
        "verbose": 0,  # Suppress detailed training output
        "gpu_ram_part": 0.95,  # Limit GPU memory usage to 90%
    }

    # Initialize the CatBoostClassifier with the current hyperparameters
    model = CatBoostClassifier(**param)

    # Create a Pool object for optimized CatBoost training
    train_pool = Pool(X_train, y_train)

    # Train the model
    model.fit(train_pool)

    # Predict on the test data
    y_test_pred = model.predict(X_test)

    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_test_pred)

    return roc_auc


# Create an Optuna study for hyperparameter optimization
study = optuna.create_study(direction="maximize")

# Optimize the study
study.optimize(objective, n_trials=10)

# Get the best parameters found by Optuna
best_params = study.best_params

# Train the final model with the best hyperparameters
best_model = CatBoostClassifier(**best_params)
best_model.fit(X_train, y_train)

# Save the best model to a file
dump(best_model, os.path.join(models_dir, "best_catboost_model_optuna.joblib"))

# Print the best parameters
print("Best parameters found: ", best_params)
# Save the classification report to a file
with open(f"{results_dir}/CatBoostBestParams.txt", "a") as file:
    file.write(f"Best Params for CatBoost\n{best_params}")

# Predict and evaluate the model on test data
y_test_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Generate and print the classification report
report = classification_report(y_test, y_test_pred)
print(report)

# Save the classification report to a file
with open(f"{results_dir}/Classification_Report.txt", "a") as file:
    file.write(f"Classification Report for CatBoost\n{report}\n{'=' * 53}\n")

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot and save ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig(f"{results_dir}/CatBoost_roc.png")
plt.close()

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

# Plot and save Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color="b", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(f"{results_dir}/CatBoost_precision_recall_curve.png")
plt.close()

# Clean up to free memory
gc.collect()
