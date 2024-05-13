import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Updated data for the models
data = {
    'Model': ['Dense Layers', 'SVM', 'XGBoost', 'Random Forest'],
    'Precision': [0.90, 0.76, 0.86, 0.88],
    'Recall': [0.88, 0.56, 0.85, 0.85],
    'F1-Score': [0.89, 0.54, 0.85, 0.86],
    'Accuracy': [0.92, 0.76, 0.89, 0.90]
}

# Create DataFrame
df_models = pd.DataFrame(data)

# Calculate a Composite Score as an average of all metrics
df_models['Composite Score'] = df_models[['Precision', 'Recall', 'F1-Score', 'Accuracy']].mean(axis=1)

# Setting the plotting style
sns.set(style="whitegrid")

# Define a list of metrics for individual plots
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Composite Score']

# Create subplots for each metric
fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 18))

for i, metric in enumerate(metrics):
    sns.barplot(x=metric, y='Model', data=df_models, palette='Set2', ax=axes[i])
    axes[i].set_title(f'Model Comparison on {metric}')
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel('Model')

# Adjust layout
plt.tight_layout()
plt.show()
