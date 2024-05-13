import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, roc_auc_score, 
                             f1_score, precision_score, recall_score, confusion_matrix, 
                             classification_report, brier_score_loss)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Predict probabilities for Test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)  # Converting probabilities to binary output

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Additional performance metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
print("Brier Score Loss:", brier_score_loss(y_test, y_pred_probs.flatten()))
print("ROC AUC:", roc_auc_score(y_test, y_pred_probs))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
