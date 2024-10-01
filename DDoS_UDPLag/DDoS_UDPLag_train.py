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

# Directories
results_dir = os.path.join("Results", "Train_Result")
model_dir = os.path.join("Models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load pre-trained scaler and encoder
scaler = load("../standard_scaler.joblib")
encoder = load("encoder(udplag).joblib")

# Load the dataset
df = pd.read_csv("../Training_datasets/UDPLag.csv")

class_to_remove = "WebDDoS"  # Change this to the class you want to remove
df = df[df["Label"] != class_to_remove]

# Encode labels
df["Label"] = encoder.transform(df["Label"])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))  # To check the encoding

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

# X,y = smote.fit_resample(X,y)
# np.save("X_train.npy",X)
# np.save("y_train.npy" , y)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
# Clean up to free memory
del X, y
gc.collect()

# Initialize classifiers with parallel processing and GPU support where applicable
models = {
    "catboost": CatBoostClassifier(
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

# Train and save models
for model_name, model in models.items():
    print(f"Training {model_name}")

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Save the trained model to disk
    dump(model, f"Models/{model_name}_model.joblib")
    print(f"{model_name} model trained and saved as {model_name}_model.joblib")

    # Clean up to free memory
    del model
    gc.collect()

# Evaluate the models
for model_name in models.keys():
    # Load the model from file
    loaded_model = load(f"Models/{model_name}_model.joblib")

    # Make predictions on the test set
    y_pred = loaded_model.predict(X_test)
    y_proba = loaded_model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Generate and print the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the classification report to a file
    with open(f"{results_dir}/Classification_Report.txt", "a") as file:
        file.write(f"Classification Report for {model_name}\n{report}\n{'=' * 53}\n")

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
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
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # Plot and save Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{results_dir}/{model_name}_precision_recall_curve.png")
    plt.close()

    # Clean up to free memory
    del loaded_model
    gc.collect()
