#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

# Load the data
df = pd.read_csv('Processed Datasets/DrDoS_DNS.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Display class distribution
class_counts = df['Label'].value_counts()
print(class_counts)

# Display the shape of the dataframe
print(df.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=27)
smote_X_train, smote_y_train = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = MaxAbsScaler()
smote_X_train = scaler.fit_transform(smote_X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(smote_X_train, smote_y_train)
y_pred = model.predict(X_test)

# Display predictions
print(y_pred)

# Print classification report
print(classification_report(y_test, y_pred))

# Predict probabilities for the positive class (class 1)
y_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

# Plot the Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Display class distribution of SMOTE training data
from collections import Counter
print(Counter(smote_y_train))
