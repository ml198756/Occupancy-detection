import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv(r'C:\Users\Administrator\Desktop\training\room_occupancy_detection_data.csv')

# 打印列名以确认
print("Columns in the dataset:", data.columns)

# 检查'DateTime'列是否存在并删除
if 'DateTime' in data.columns:
    print("Dropping 'DateTime' column.")
    data = data.drop('DateTime', axis=1)
else:
    print("Column 'DateTime' not found in the dataset. Please check the CSV file.")

# 确认所有列类型
print("Column types before encoding:", data.dtypes)

# 处理分类特征
encoder = OneHotEncoder()
room_type_encoded = encoder.fit_transform(data[['room_type']]).toarray()
room_type_encoded = pd.DataFrame(room_type_encoded, columns=encoder.get_feature_names_out(['room_type']))
data = pd.concat([data.drop('room_type', axis=1), room_type_encoded], axis=1)

# 确认所有列类型
print("Column types after encoding:", data.dtypes)

# 打印数据样本
print("Data sample after preprocessing:\n", data.head())

# 确认是否有非数值列
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Non-numeric column found: {col}")

X = data.drop('occupancy_ground_truth', axis=1)
y = data['occupancy_ground_truth']

# 分割数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 标准化数值特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
