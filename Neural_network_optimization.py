import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             f1_score, precision_score, recall_score, confusion_matrix,
                             classification_report, brier_score_loss, make_scorer)
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = pd.read_csv(r'C:\Users\Administrator\Desktop\training\room_occupancy_detection_data.csv')

# Drop 'DateTime' column if it exists
if 'DateTime' in data.columns:
    data = data.drop('DateTime', axis=1)

# Process categorical feature
encoder = OneHotEncoder()
room_type_encoded = encoder.fit_transform(data[['room_type']]).toarray()
room_type_encoded = pd.DataFrame(room_type_encoded, columns=encoder.get_feature_names_out(['room_type']))
data = pd.concat([data.drop('room_type', axis=1), room_type_encoded], axis=1)

# Ensure all columns are numeric
X = data.drop('occupancy_ground_truth', axis=1)
y = data['occupancy_ground_truth']

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

class OccupancyDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(OccupancyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.output = nn.Linear(hidden_dims[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2, lr=0.001, weight_decay=0):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = OccupancyDetectionModel(input_dim, hidden_dims, dropout_rate).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, X, y, num_epochs=10, batch_size=32):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        for epoch in range(num_epochs):
            self.model.train()
            permutation = torch.randperm(X.size()[0])

            for i in range(0, X.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X[indices], y[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            # Clear GPU cache to avoid memory issues
            torch.cuda.empty_cache()

        return self

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = self.model(X).squeeze().detach().cpu().numpy()
        return np.vstack((1 - outputs, outputs)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

# Define the parameter grid
param_grid = {
    'hidden_dims': [(128, 64, 32), (256, 128, 64), (128, 128, 64)],
    'dropout_rate': [0.2, 0.3, 0.4],
    'lr': [0.001, 0.0005, 0.0001],
    'weight_decay': [0, 1e-4, 1e-3]
}

# Create the scorer
scorer = make_scorer(balanced_accuracy_score)

# Create the RandomizedSearchCV object
input_dim = X_train.shape[1]
pytorch_classifier = PyTorchClassifier(input_dim=input_dim)
random_search = RandomizedSearchCV(pytorch_classifier, param_distributions=param_grid, n_iter=10, scoring=scorer, cv=3, verbose=1)

# Perform random search
random_search.fit(X_train.cpu().numpy(), y_train.cpu().numpy(), num_epochs=50, batch_size=64)

# Best parameters and model evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test.cpu().numpy())

# Print the results
print('Best parameters found:', random_search.best_params_)
print('Test accuracy:', (y_pred == y_test.cpu().numpy()).mean())
print("Balanced Accuracy:", balanced_accuracy_score(y_test.cpu().numpy(), y_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test.cpu().numpy(), y_pred))
print("Brier Score Loss:", brier_score_loss(y_test.cpu().numpy(), y_pred))
print("ROC AUC:", roc_auc_score(y_test.cpu().numpy(), y_pred))
print("F1 Score:", f1_score(y_test.cpu().numpy(), y_pred))
print("Precision:", precision_score(y_test.cpu().numpy(), y_pred))
print("Recall:", recall_score(y_test.cpu().numpy(), y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test.cpu().numpy(), y_pred))
print(classification_report(y_test.cpu().numpy(), y_pred))
