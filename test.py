import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import optuna
import pandas as pd

# 定义加载数据的自定义 Dataset 类
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = np.expand_dims(x, axis=-1)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 加载数据
def load_data():
    data = []
    labels = []
    label_map = {"N": 0, "A": 1, "V": 2, "L": 3, "R": 4}

    for label in label_map.keys():
        file_path = f'./assets/{label}.npy'
        if os.path.exists(file_path):
            signals = np.load(file_path)
            data.append(signals)
            labels.append(np.full(signals.shape[0], label_map[label]))

    if data:
        data = np.concatenate(data)
        labels = np.concatenate(labels)
    else:
        raise ValueError("No data found.")

    return data, labels

# Transformer 模型定义
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_heads, num_layers, hidden_dim):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# 训练模型函数
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    logs = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        logs["train_loss"].append(avg_train_loss)
        logs["val_loss"].append(avg_val_loss)
        logs["val_accuracy"].append(val_acc)

        # 检查验证损失是否改善
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break

    return logs

# 定义 Optuna 目标函数
def objective(trial):
    data, labels = load_data()

    # 缩小后的超参数搜索空间
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)  # 使用 suggest_float 代替 suggest_loguniform
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    num_epochs = 20
    patience = 2

    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(input_size=1, num_classes=len(np.unique(labels)),
                                  num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)

    logs = train_model(model, train_loader, val_loader, num_epochs=num_epochs,
                       learning_rate=learning_rate, patience=patience)

    # 使用 step 关键字替换 epoch
    final_val_accuracy = logs["val_accuracy"][-1]
    trial.report(final_val_accuracy, step=num_epochs)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return final_val_accuracy

# 使用 MedianPruner 进行自动调参和提前停止
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)

# 可视化结果
def plot_results(study):
    for i, trial in enumerate(study.trials):
        if "val_accuracy" in trial.user_attrs:
            plt.plot(range(len(trial.user_attrs["val_accuracy"])), trial.user_attrs["val_accuracy"], label=f"Trial_{i + 1}")
    plt.title("Validation Accuracy for Different Hyperparameters (Optuna)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# 提取结果并保存
results_df = pd.DataFrame([(trial.number, trial.params, trial.value) for trial in study.trials], columns=["Trial", "Params", "Final Validation Accuracy"])
results_df.to_csv("optuna_results.csv", index=False)
print("\nResults Summary saved to optuna_results.csv")
print(results_df)

# 绘图
plot_results(study)

# 找出表现最佳的超参数组合
best_trial = study.best_trial
print("\nBest Trial:")
print(f"Trial Number: {best_trial.number}")
print(f"Best Validation Accuracy: {best_trial.value}")
print("Best Hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")
