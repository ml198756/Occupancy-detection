import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, matthews_corrcoef, log_loss, roc_auc_score, \
    f1_score, precision_score, recall_score, confusion_matrix, classification_report

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

# 定义随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数的模型进行预测和评估
best_rf = grid_search.best_estimator_

# 在测试集上评估
y_test_pred = best_rf.predict(X_test)
y_test_pred_proba = best_rf.predict_proba(X_test)

# 测试集的评估指标
print("Test Set Evaluation:")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_test_pred))
print("Log Loss:", log_loss(y_test, y_test_pred_proba))
print("ROC AUC:", roc_auc_score(pd.get_dummies(y_test), y_test_pred_proba, multi_class='ovr'))
print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
