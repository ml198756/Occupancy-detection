##############################################################################
### PACKAGES ###
##############################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

##############################################################################
### LOAD DATA ###
##############################################################################

# Load the data
dataset = pd.read_csv('/Users/mulin/Downloads/room_occupancy_detection_data.csv')
dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%d-%m-%y %H:%M')
full_dataset = dataset.copy()

##############################################################################
### DATA SPLITTING ###
##############################################################################

# Defining input features and target variable
X = full_dataset[['indoor_co2_concentration', 'indoor_operative_temperature', 'indoor_relative_humidity', 'room_type', 'room_number', 'floor_area']]
y = full_dataset['occupancy_ground_truth']

# Splitting the data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # First split, 60% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Second split, equally divide the remaining 40% into 20% each

##############################################################################
### MODEL TRAINING AND EVALUATION WITH HYPERPARAMETER OPTIMIZATION ###
##############################################################################

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

# Initialize the SVM model
svm = SVC(probability=True)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best estimator to make predictions
y_val_pred = grid_search.predict(X_val)
y_val_pred_proba = grid_search.predict_proba(X_val)[:, 1]

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

# Similarly, evaluate on test set
y_test_pred = grid_search.predict(X_test)
y_test_pred_proba = grid_search.predict_proba(X_test)[:, 1]

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

##############################################################################
### SCRIPT END ###
##############################################################################
