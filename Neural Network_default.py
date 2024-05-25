import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, roc_auc_score, 
                             f1_score, precision_score, recall_score, confusion_matrix, 
                             classification_report, brier_score_loss)

# Load the data
data = pd.read_csv('/Users/mulin/Downloads/room_occupancy_detection_data.csv')

# Convert datetime column with explicit format
data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%y %H:%M')
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data = data.drop('datetime', axis=1)  # Remove the original datetime column

# Example: Assume 'room_type' is a categorical column that needs encoding
encoder = OneHotEncoder()
room_type_encoded = encoder.fit_transform(data[['room_type']]).toarray()
room_type_encoded = pd.DataFrame(room_type_encoded, columns=encoder.get_feature_names_out(['room_type']))
data = pd.concat([data.drop('room_type', axis=1), room_type_encoded], axis=1)

# Split features and target variable
X = data.drop('occupancy_ground_truth', axis=1)
y = data['occupancy_ground_truth']

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Define the model
class OccupancyDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(OccupancyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
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

# Initialize the model, loss function and optimizer
model = OccupancyDetectionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_outputs = model(X_val).squeeze()
    val_loss = criterion(val_outputs, y_val)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Predict probabilities for the Test set
model.eval()
y_pred_probs = model(X_test).squeeze().detach().numpy()
y_pred = (y_pred_probs > 0.5).astype(int)

# Evaluate the model
test_loss = criterion(torch.tensor(y_pred_probs), y_test).item()
test_acc = (y_pred == y_test.numpy()).mean()
print('Test accuracy:', test_acc)

# Additional performance metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
print("Brier Score Loss:", brier_score_loss(y_test, y_pred_probs))
print("ROC AUC:", roc_auc_score(y_test, y_pred_probs))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
