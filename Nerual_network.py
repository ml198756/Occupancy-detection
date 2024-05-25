import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef, log_loss, roc_auc_score, \
    f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载Excel文件
file_path = 'C:/Users/Administrator/Desktop/training/数据集2/anonymizedData.xlsx'  # 更新为实际路径
data = pd.ExcelFile(file_path)

# 表单名称
floor_groups = {
    'Floor1': ['Floor1W', 'Floor1S'],
    'Floor2': ['Floor2W', 'Floor2S'],
    'Floor3': ['Floor3'],
    'Floor4': ['Floor4']
}


# 处理每个楼层的数据
def preprocess_floor_data(df):
    # 根据Groundtruth划分为三类
    def categorize_occupancy(occupancy):
        if occupancy <= 10:
            return 0  # 低占用
        elif occupancy <= 20:
            return 1  # 中等占用
        else:
            return 2  # 高占用

    df['occupancy_category'] = df['Groundtruth'].apply(categorize_occupancy)

    # 定义特征列和目标变量
    feature_columns = [col for col in df.columns if 'PIR_' in col or 'CO2_ppm_' in col or col in [
        'Inst_kW_Load_Light', 'Inst_kW_Load_Plug', 'Inst_kW_Load_Elec', 'AP1', 'AP2', 'AP3', 'AP_Total']]
    X = df[feature_columns]
    y = df['occupancy_category']

    return X, y


# 合并每层的数据
all_X = []
all_y = []
for floor, sheets in floor_groups.items():
    floor_data = []
    for sheet in sheets:
        df = data.parse(sheet)
        floor_data.append(df)
    combined_floor_data = pd.concat(floor_data, ignore_index=True)
    X, y = preprocess_floor_data(combined_floor_data)
    all_X.append(X)
    all_y.append(y)

# 合并所有楼层的数据
X_full = pd.concat(all_X, axis=0)
y_full = pd.concat(all_y, axis=0)

# 缺失值填补
imputer = SimpleImputer(strategy='mean')
X_full = imputer.fit_transform(X_full)

# 特征缩放
scaler = StandardScaler()
X_full = scaler.fit_transform(X_full)

# 拆分数据集为训练集（60%），验证集（20%）和测试集（20%）
X_train, X_temp, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.4, random_state=42)  # 第一次拆分，60%用于训练
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 第二次拆分，剩余的40%平均分成20%各自

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)


# 定义 PyTorch 模型
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2):
        super(NeuralNet, self).__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    model = model.to(device)
    best_model = None
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_model = model.state_dict()
    model.load_state_dict(best_model)
    return model


# 定义超参数网格
param_grid = {
    'hidden_dims': [(128, 64, 32), (256, 128, 64), (128, 128, 64)],
    'dropout_rate': [0.2, 0.3, 0.4],
    'lr': [0.001, 0.0005, 0.0001],
    'weight_decay': [0, 1e-4, 1e-3]
}

# 手动实现网格搜索
best_params = None
best_acc = 0
for hidden_dims, dropout_rate, lr, weight_decay in product(param_grid['hidden_dims'], param_grid['dropout_rate'],
                                                           param_grid['lr'], param_grid['weight_decay']):
    model = NeuralNet(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        best_params = (hidden_dims, dropout_rate, lr, weight_decay)

print("Best parameters found: ", best_params)

# 使用最佳参数训练最终模型
hidden_dims, dropout_rate, lr, weight_decay = best_params
best_model = NeuralNet(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=lr, weight_decay=weight_decay)
best_model = train_model(best_model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# 在测试集上评估
best_model.eval()
y_test_pred_proba = []
y_test_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = best_model(X_batch)
        y_test_pred_proba.extend(outputs.cpu().numpy())
        y_test_true.extend(y_batch.cpu().numpy())
y_test_pred_proba = np.array(y_test_pred_proba)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)
y_test_true = np.array(y_test_true)

# 测试集的评估指标
print("Test Set Evaluation:")
print("Balanced Accuracy:", balanced_accuracy_score(y_test_true, y_test_pred))
print("Accuracy:", accuracy_score(y_test_true, y_test_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test_true, y_test_pred))
print("Log Loss:", log_loss(y_test_true, y_test_pred_proba))
print("ROC AUC:", roc_auc_score(pd.get_dummies(y_test_true), y_test_pred_proba, multi_class='ovr'))
print("F1 Score:", f1_score(y_test_true, y_test_pred, average='weighted'))
print("Precision:", precision_score(y_test_true, y_test_pred, average='weighted'))
print("Recall:", recall_score(y_test_true, y_test_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_true, y_test_pred))
print(classification_report(y_test_true, y_test_pred))
