import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from joblib import dump, parallel_backend
import gc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

""" Training Datasets """

train_df = pd.read_csv("Training_Datasets/Syn.csv")


train_df["Label"] = encoder.fit_transform(train_df["Label"])

# Prepare features and labels
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

del train_df
gc.collect()

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=27)
X_smote, y_train = smote.fit_resample(X, y)

del X
del y
gc.collect()

# Scale features
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_smote)

del X_smote
gc.collect()

""" Testing Datasets """

test_df = pd.read_csv("Testing_datasets/Syn.csv").drop(
    columns=[
        "Unnamed: 0",
        " Timestamp",
        "SimillarHTTP",
        "Flow ID",
        " Source IP",
        " Destination IP",
    ]
)
test_df.columns = test_df.columns.str.strip()

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df = test_df.dropna()

test_df["Label"] = label_encoder.transform(test_df["Label"])

# Prepare features and labels
X = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
X_test = scaler.transform(X)

del test_df, X
gc.collect()


# Define the objective function to optimize
def optimize_catboost(trial):
    # Hyperparameters to tune
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int(
            "depth", 3, 7
        ),  # Smaller depth for less memory usage
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5.0),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Plain", "Ordered"]
        ),
        "task_type": "GPU",
        "devices": "0",  # Specify GPU
        "gpu_ram_part": 0.9,  # Use 90% of GPU RAM to avoid overloading
    }

    # Create a CatBoost classifier
    model = CatBoostClassifier(**params)

    # Train the model on the training set with early stopping to reduce overfitting
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=False,
        early_stopping_rounds=50,
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Free up memory
    del model
    gc.collect()

    return f1


study = optuna.create_study(direction="maximize")
study.optimize(optimize_catboost, n_trials=5)

# Print the best hyperparameters and the corresponding F1 score
best_params = study.best_trial.params
best_f1 = study.best_trial.value
print(f"Best hyperparameters: {best_params}")
print(f"Best F1 score: {best_f1:.4f}")
