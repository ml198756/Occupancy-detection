import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
dataset_path = r'C:\Users\Administrator\Desktop\training\room_occupancy_detection_data.csv'
dataset = pd.read_csv(dataset_path)

# Remove 'DateTime' column if it exists
dataset = dataset.drop(columns=['DateTime'], errors='ignore')

# Encode categorical feature 'room_type'
encoder = OneHotEncoder()
room_type_encoded = encoder.fit_transform(dataset[['room_type']]).toarray()
room_type_encoded = pd.DataFrame(room_type_encoded, columns=encoder.get_feature_names_out(['room_type']))
dataset = pd.concat([dataset.drop('room_type', axis=1), room_type_encoded], axis=1)

# Define input features and target variable
X = dataset.drop('occupancy_ground_truth', axis=1)
y = dataset['occupancy_ground_truth']

# Split the data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best estimator to make predictions
best_rf = grid_search.best_estimator_
y_val_pred = best_rf.predict(X_val)
y_val_pred_proba = best_rf.predict_proba(X_val)[:, 1]

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
y_test_pred = best_rf.predict(X_test)
y_test_pred_proba = best_rf.predict_proba(X_test)[:, 1]

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
