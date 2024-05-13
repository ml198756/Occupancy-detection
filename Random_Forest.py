

#%%
##############################################################################
### PACKAGES ###
##############################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

#%%
##############################################################################
### LOAD DATA ###
##############################################################################

# make to input dataset right
dataset = pd.read_csv('/Users/mulin/Downloads/room_occupancy_detection_data.csv')
dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%d-%m-%y %H:%M')
full_dataset = dataset.copy()

#%%
##############################################################################
### DATA SPLITTING ###
##############################################################################

# Defining input features and target variable
X = full_dataset[['indoor_co2_concentration', 'indoor_operative_temperature', 'indoor_relative_humidity', 'room_type', 'room_number', 'floor_area']]
y = full_dataset['occupancy_ground_truth']

# Splitting the data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # First split, 60% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Second split, equally divide the remaining 40% into 20% each

#%%
##############################################################################
### MODEL TRAINING AND EVALUATION ###
##############################################################################

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = rf.predict(X_val)
y_val_pred_proba = rf.predict_proba(X_val)[:, 1]

# Evaluation metrics for validation set
print("Validation Set Evaluation:")
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_val_pred))
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_val, y_val_pred))
print("Brier Score Loss:", brier_score_loss(y_val, y_val_pred_proba))
print("ROC AUC:", roc_auc_score(y_val, y_val_pred_proba))
print("F1 Score:", f1_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = rf.predict(X_test)
y_test_pred_proba = rf.predict_proba(X_test)[:, 1]

# Evaluation metrics for test set
print("Test Set Evaluation:")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_test_pred))
print("Brier Score Loss:", brier_score_loss(y_test, y_test_pred_proba))
print("ROC AUC:", roc_auc_score(y_test, y_test_pred_proba))
print("F1 Score:", f1_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


#%%
##############################################################################
### SCRIPT END ###
##############################################################################
