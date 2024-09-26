import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from joblib import dump, load
import tempfile

# Directories
model_dir = os.path.join("Models")
os.makedirs(model_dir, exist_ok=True)

# Load encoder and scaler
encoder = LabelEncoder()
scaler = load(os.path.join("..", "standard_scaler.joblib"))

# Initialize classifiers
def initialize_classifiers():
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

# Process and train model in batches
def process_batches(models):
    for model_name, model in models.items():
        X_train = np.load("Combined_Data/X_train.npy")
        y_train = np.load("Combined_Data/y_train.npy")

        model.fit(X_train,y_train)

        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        dump(model, model_path)
        print(f"{model_name} model trained and saved at {model_path}")

        del model
        gc.collect()

# Main function
def main():
    models = initialize_classifiers()
    process_batches(models)

if __name__ == "__main__":
    main()
