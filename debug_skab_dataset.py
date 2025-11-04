import os
import numpy as np
import pandas as pd
import yaml
from data.datasets import DataManager, SKABDataset

# --------------------------
# 1. 加载配置
# --------------------------
with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

data_manager = DataManager(config)
skab_data_path = os.path.join(config['experiment']['data_root'], config['datasets']['SKAB']['path'])
window_size = config['data']['window_size']
stride = config['data']['stride']
print(f"当前窗口大小: {window_size}, 步长: {stride}")


# --------------------------
# 2. 一次性加载训练集原始数据（窗口化前）
#    确保后续窗口化和检查使用同一批数据
# --------------------------
print("\n【加载训练集原始数据（唯一来源）】")
# 加载原始数据（未窗口化）
train_raw = SKABDataset(
    data_path=skab_data_path,
    window_size=window_size,
    stride=stride,
    split='train',
    normalize=False
)
train_raw_data = train_raw.data  # 原始特征数据
train_raw_labels = train_raw.labels  # 原始标签（关键：窗口化和检查共用此标签）
print(f"训练集原始数据样本数：{len(train_raw_labels)}")
print(f"训练集原始标签分布：{np.unique(train_raw_labels, return_counts=True)}")


# --------------------------
# 3. 基于同一批原始数据创建窗口化数据集
#    避免两次加载导致的差异
# --------------------------
print("\n【基于原始数据创建窗口化数据集】")
# 手动创建窗口化样本（复用BaseTimeSeriesDataset的逻辑，但使用train_raw的原始数据）
class TempWindowDataset:
    def __init__(self, raw_data, raw_labels, window_size, stride):
        self.raw_data = raw_data
        self.raw_labels = raw_labels
        self.window_size = window_size
        self.stride = stride
        self.samples = self._create_windows()
    
    def _create_windows(self):
        samples = []
        n_samples = (len(self.raw_data) - self.window_size) // self.stride + 1
        for i in range(n_samples):
            start = i * self.stride
            end = start + self.window_size
            window_data = self.raw_data[start:end]
            window_labels = self.raw_labels[start:end]
            # 窗口标签逻辑：含1则为1，全0则为0
            label = 1 if np.any(window_labels == 1) else 0
            samples.append((window_data, label))
        return samples

# 使用同一批原始数据创建窗口化数据集
train_dataset = TempWindowDataset(
    raw_data=train_raw_data,
    raw_labels=train_raw_labels,
    window_size=window_size,
    stride=stride
)
print(f"窗口化后样本数：{len(train_dataset.samples)}")


# --------------------------
# 4. 检查窗口化逻辑（基于同一批数据）
# --------------------------
print("\n【窗口化逻辑验证（同一批原始数据）】")
if 0 in train_raw_labels:
    # 直接从窗口化样本的实际起始索引中寻找全正常窗口（而非从normal_indices中找）
    valid_samples = []
    for i in range(len(train_dataset.samples)):
        start = i * stride  # 窗口化样本i的实际起始索引
        end = start + window_size
        if end > len(train_raw_labels):
            continue
        window_raw = train_raw_labels[start:end]
        if np.all(window_raw == 0):  # 实际窗口全为0
            valid_samples.append(i)
            if len(valid_samples) >= 10:
                break
    if valid_samples:
        print(f"找到{len(valid_samples)}个全正常的实际窗口，验证标签：")
        for i in valid_samples[:5]:
            start = i * stride
            end = start + window_size
            window_data, window_label = train_dataset.samples[i]
            actual_window_labels = train_raw_labels[start:end]
            print(f"实际窗口[{start}:{end}]："
                  f"实际标签全为0？{np.all(actual_window_labels == 0)}，"
                  f"窗口化后标签：{window_label}")
    else:
        print("未找到全正常的实际窗口，需进一步减小窗口大小")
else:
    print("原始数据中无正常样本，回到数据加载阶段排查")


# --------------------------
# 5. 统计窗口化后的标签分布
# --------------------------
print("\n【窗口化后标签分布】")
train_labels = np.array([label for _, label in train_dataset.samples])
print(f"训练集窗口化后：总样本{len(train_labels)}，"
      f"异常{np.sum(train_labels)}，正常{len(train_labels)-np.sum(train_labels)}")
print(f"异常比例：{np.mean(train_labels):.4f}")
print("标签分布：", np.unique(train_labels, return_counts=True))


# --------------------------
# 6. 检查文件是否重复加载
# --------------------------
print("\n【文件去重检查】")
# 复用SKABDataset的加载逻辑，统计实际加载的文件数
class FileCheckSKAB(SKABDataset):
    def __init__(self, data_path):
        self.loaded_files = set()
        self._load_files(data_path)
    
    def _load_files(self, data_path):
        for subdir in ['valve1', 'valve2', 'other', 'anomaly-free']:
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.exists(subdir_path):
                continue
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            for file in csv_files:
                file_path = os.path.join(subdir_path, file)
                self.loaded_files.add(file_path)
        print(f"实际加载的唯一文件数：{len(self.loaded_files)}")
        # 打印前5个文件路径，确认无重复
        print("部分加载的文件路径：", list(self.loaded_files)[:5])

# 检查文件去重是否生效
file_checker = FileCheckSKAB(skab_data_path)


# --------------------------
# 7. 原始CSV文件统计（验证数据源）
# --------------------------
print("\n【原始CSV文件总统计】")
subdirs = ['valve1', 'valve2', 'other', 'anomaly-free']
total_all = 0
anomaly_all = 0
normal_all = 0
for subdir in subdirs:
    subdir_path = os.path.join(skab_data_path, subdir)
    if not os.path.exists(subdir_path):
        print(f"子目录不存在：{subdir_path}")
        continue
    csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
    total_sub = 0
    anomaly_sub = 0
    normal_sub = 0
    for file in csv_files:
        file_path = os.path.join(subdir_path, file)
        try:
            df = pd.read_csv(file_path, sep=';')
        except Exception as e:
            print(f"读取失败 {file_path}：{e}")
            continue
        n_samples = len(df)
        if 'anomaly' in df.columns:
            anomaly = df['anomaly'].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
        else:
            anomaly = 0  # anomaly-free目录无anomaly列，全为正常
        normal = n_samples - anomaly
        total_sub += n_samples
        anomaly_sub += anomaly
        normal_sub += normal
    print(f"子目录 {subdir}：总样本{total_sub}，异常{anomaly_sub}，正常{normal_sub}")
    total_all += total_sub
    anomaly_all += anomaly_sub
    normal_all += normal_sub
print(f"原始数据总样本：{total_all}，异常{anomaly_all}，正常{normal_all}，异常比例{anomaly_all/total_all:.4f}")