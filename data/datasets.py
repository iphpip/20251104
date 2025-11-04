import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import wfdb
import json
from typing import Dict, List, Tuple, Optional, Union
from glob import glob
from tqdm import tqdm
from typing import Tuple

class BaseTimeSeriesDataset(Dataset):
    """时间序列数据集基类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 window_size: int = 100, stride: int = 1, normalize: bool = True):
        self.data = self._normalize_data(data) if normalize else data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.samples = self._create_samples()
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """标准化数据（避免不同尺度特征影响模型）"""
        scaler = StandardScaler()
        # 处理单变量/多变量数据
        if len(data.shape) == 1:
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            return scaler.fit_transform(data)
    
    def _create_samples(self) -> List[Tuple[np.ndarray, int]]:
        """创建滑动窗口样本 - 优化版本，减少异常窗口污染"""
        samples = []
        n_samples = (len(self.data) - self.window_size) // self.stride + 1
        
        # 统计原始数据中连续正常段落的长度
        normal_segments = []
        current_segment_start = 0
        in_normal_segment = False
        
        for i in range(len(self.labels)):
            if self.labels[i] == 0:  # 正常点
                if not in_normal_segment:
                    in_normal_segment = True
                    current_segment_start = i
            else:  # 异常点
                if in_normal_segment:
                    normal_segments.append((current_segment_start, i-1))
                    in_normal_segment = False
        # 处理最后一个段落
        if in_normal_segment:
            normal_segments.append((current_segment_start, len(self.labels)-1))
        
        print(f"发现 {len(normal_segments)} 个连续正常段落")
        
        # 优先从长的连续正常段落中采样
        for start, end in normal_segments:
            segment_length = end - start + 1
            if segment_length >= self.window_size:
                # 从这个段落中可以采样的窗口数
                n_segment_samples = (segment_length - self.window_size) // self.stride + 1
                for i in range(n_segment_samples):
                    start_idx = start + i * self.stride
                    end_idx = start_idx + self.window_size
                    if end_idx <= end + 1:  # 确保不越界
                        window_data = self.data[start_idx:end_idx]
                        samples.append((window_data, 0))  # 正常窗口
        
        # 如果正常窗口太少，补充一些混合窗口
        target_normal_ratio = 0.3  # 目标正常窗口比例
        target_normal_count = int(target_normal_ratio * n_samples)
        
        if len(samples) < target_normal_count:
            # 从剩余位置采样一些窗口
            remaining_indices = []
            for i in range(n_samples):
                start_idx = i * self.stride
                end_idx = start_idx + self.window_size
                # 只考虑异常点比例较低的窗口
                anomaly_count = np.sum(self.labels[start_idx:end_idx] == 1)
                if anomaly_count <= 1:  # 最多允许1个异常点
                    remaining_indices.append(i)
            
            # 随机选择一些补充窗口
            import random
            needed_count = target_normal_count - len(samples)
            if needed_count > 0 and remaining_indices:
                selected_indices = random.sample(remaining_indices, min(needed_count, len(remaining_indices)))
                for idx in selected_indices:
                    start_idx = idx * self.stride
                    end_idx = start_idx + self.window_size
                    window_data = self.data[start_idx:end_idx]
                    window_labels = self.labels[start_idx:end_idx]
                    label = 1 if np.any(window_labels == 1) else 0
                    samples.append((window_data, label))
        
        # 最后添加剩余的异常窗口
        all_indices = set(range(n_samples))
        used_indices = set()
        # 找出已使用的索引
        for i in range(n_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]
            # 检查这个窗口是否已经在samples中
            for sample_data, _ in samples:
                if np.array_equal(window_data, sample_data):
                    used_indices.add(i)
                    break
        
        remaining_indices = list(all_indices - used_indices)
        for i in remaining_indices:
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]
            window_labels = self.labels[start_idx:end_idx]
            label = 1 if np.any(window_labels == 1) else 0
            samples.append((window_data, label))
        
        print(f"最终样本分布: 总样本{len(samples)}, 正常{sum(1 for _, label in samples if label == 0)}, 异常{sum(1 for _, label in samples if label == 1)}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        # 强制转换为float32，避免numpy.object_类型错误
        return torch.FloatTensor(np.asarray(data, dtype=np.float32)), torch.tensor(label, dtype=torch.long)

class NABDataset(BaseTimeSeriesDataset):
    """NAB数据集加载器（修复标签匹配+移除进度条，优化重复加载问题）"""
    
    # 类变量：存储全量数据（只加载一次，所有实例复用）
    _full_data = None
    _full_labels = None
    # 类变量：缓存数据路径（确保不同实例使用相同路径，避免路径不一致导致的问题）
    _cached_data_path = None
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train', normalize: bool = True):
        self.data_path = data_path.rstrip(os.sep)
        # 1. 检查全量数据是否已加载，且数据路径一致（避免不同路径混用）
        if (NABDataset._full_data is None or NABDataset._full_labels is None) or (NABDataset._cached_data_path != self.data_path):
            # 若未加载或路径变更，重新加载全量数据，并缓存路径
            NABDataset._full_data, NABDataset._full_labels = self._load_full_nab_data()
            NABDataset._cached_data_path = self.data_path
        
        # 2. 从全量数据中划分当前split的子集（不重复加载）
        data, labels = self._split_data(split)
        super().__init__(data, labels, window_size, stride, normalize)
    
    def _load_full_nab_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载全量NAB数据（只执行一次）：包含标签读取、CSV遍历、数据处理与合并"""
        # 1. 加载标签文件
        label_file = os.path.join(self.data_path, "labels", "combined_windows.json")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"标签文件不存在：{label_file}")
        
        with open(label_file, 'r') as f:
            anomaly_windows = json.load(f)  # 标签键格式："artificialWithAnomaly/art_daily_flatmiddle.csv"
        
        # 2. 遍历所有CSV文件（移除tqdm进度条）
        csv_files = glob(
            os.path.join(self.data_path, "data", "**", "*.csv"),  # 明确从data子目录搜索
            recursive=True
        )
        # 过滤无效文件（系统文件、无关目录）
        csv_files = [
            f for f in csv_files 
            if "__MACOSX" not in f and not os.path.basename(f).startswith('.')
        ]
        
        if not csv_files:
            raise ValueError(f"在 {os.path.join(self.data_path, 'data')} 下未找到CSV文件")
        
        all_data = []
        all_labels = []
        
        # 遍历文件（无进度条）
        for file_path in csv_files:
            # 计算文件相对路径（关键：与标签键匹配，移除"data/"前缀）
            rel_path = os.path.relpath(file_path, self.data_path)  # 结果：data/xxx/xxx.csv
            file_key = rel_path.replace(os.sep, '/').replace("data/", "", 1)  # 结果：xxx/xxx.csv
            
            # 读取数据
            try:
                df = pd.read_csv(
                    file_path,
                    parse_dates=['timestamp']
                )
            except Exception as e:
                print(f"跳过 {os.path.basename(file_path)}：{str(e)}")
                continue
            
            # 检查必要列
            required_cols = ['timestamp', 'value']
            if not set(required_cols).issubset(df.columns):
                print(f"跳过 {os.path.basename(file_path)}：缺少必要列{required_cols}")
                continue
            
            # 提取特征列（保持二维结构）
            data = df['value'].values.reshape(-1, 1)
            n_samples = len(data)
            if n_samples == 0:
                print(f"跳过 {os.path.basename(file_path)}：空文件")
                continue
            
            # 生成标签（默认全0：正常）
            labels = np.zeros(n_samples, dtype=int)
            
            # 特殊处理：artificialNoAnomaly目录的文件强制为正常（不标记异常）
            if "artificialNoAnomaly" in file_key:
                print(f"加载无异常文件 {os.path.basename(file_path)}：总样本{n_samples}")
            else:
                # 从标签文件匹配并标记异常区间
                if file_key in anomaly_windows:
                    timestamps = pd.to_datetime(df['timestamp'])
                    
                    for window in anomaly_windows[file_key]:
                        try:
                            window_start = pd.to_datetime(window[0])
                            window_end = pd.to_datetime(window[1])
                        except (ValueError, TypeError) as e:
                            print(f"警告：{os.path.basename(file_path)} 的窗口 {window} 格式错误 -> {str(e)}")
                            continue
                        
                        # 标记异常区间
                        mask = (timestamps >= window_start) & (timestamps <= window_end)
                        if np.any(mask):
                            labels[mask] = 1  # 异常样本标记为1
                
                # 统计当前文件的样本分布
                n_anomaly = np.sum(labels)
                n_normal = n_samples - n_anomaly
                print(f"加载文件 {os.path.basename(file_path)}：总样本{n_samples}，异常{n_anomaly}，正常{n_normal}")
            
            all_data.append(data)
            all_labels.append(labels)
        
        # 合并全量数据
        if not all_data:
            raise ValueError("未加载到有效数据")
        full_data = np.vstack(all_data)
        full_labels = np.concatenate(all_labels)
        print(f"\n合并后NAB全量数据：总样本{len(full_labels)}，异常样本{np.sum(full_labels)}，异常比例{np.sum(full_labels)/len(full_labels):.4f}")
        
        return full_data, full_labels
    
    def _split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """从全量数据中划分指定split的子集（仅切分，不重复加载）"""
        n_total = len(NABDataset._full_labels)
        n_train = int(0.7 * n_total)
        n_val = int(0.8 * n_total)
        
        # 按split返回对应数据
        if split == 'train':
            split_data = NABDataset._full_data[:n_train]
            split_labels = NABDataset._full_labels[:n_train]
            print(f"\n{split}集：样本数{len(split_labels)}（占全量70%）")
        elif split == 'test':
            split_data = NABDataset._full_data[n_val:]
            split_labels = NABDataset._full_labels[n_val:]
            print(f"\n{split}集：样本数{len(split_labels)}（占全量20%）")
        else:  # val集
            split_data = NABDataset._full_data[n_train:n_val]
            split_labels = NABDataset._full_labels[n_train:n_val]
            print(f"\n{split}集：样本数{len(split_labels)}（占全量10%）")
        
        return split_data, split_labels

class SKABDataset(BaseTimeSeriesDataset):
    """SKAB数据集加载器"""
    
    # 类变量：存储全量数据（加载一次后复用）
    _full_data = None
    _full_labels = None
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train', normalize: bool = True):
        # 若全量数据未加载，则加载一次；否则直接复用
        if SKABDataset._full_data is None or SKABDataset._full_labels is None:
            SKABDataset._full_data, SKABDataset._full_labels = self._load_full_skab_data(data_path)
        
        # 从全量数据中划分当前split的数据
        data, labels = self._split_data(SKABDataset._full_data, SKABDataset._full_labels, split)
        super().__init__(data, labels, window_size, stride, normalize)
    
    def _load_full_skab_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载并处理全量数据"""
        loaded_files = set()
        normal_data = []
        normal_labels = []
        anomaly_data = []
        anomaly_labels = []
        
        print("\n===== 开始加载文件并提取样本=====")
        # （以下逻辑与你原来的_load_skab_data中“加载+合并+整体打乱”部分完全一致）
        # 1. 循环处理子目录和文件（提取正常/异常样本）
        for subdir in ['valve1', 'valve2', 'other', 'anomaly-free']:
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.exists(subdir_path):
                print(f"警告：SKAB子目录 {subdir_path} 不存在，跳过")
                continue
                
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            if not csv_files:
                print(f"子目录 {subdir} 无CSV文件，跳过")
                continue
            
            print(f"\n处理子目录 {subdir}（{len(csv_files)}个文件）：")
            for file in csv_files:
                file_path = os.path.join(subdir_path, file)
                if file_path in loaded_files:
                    print(f"  跳过重复文件：{file}")
                    continue
                loaded_files.add(file_path)
                
                try:
                    df = pd.read_csv(file_path, sep=';')
                except Exception as e:
                    print(f"  读取文件 {file} 失败：{e}，跳过")
                    continue
                
                # 提取特征列
                if subdir == 'anomaly-free':
                    data_columns = [col for col in df.columns if col != 'datetime']
                else:
                    data_columns = [col for col in df.columns if col not in ['anomaly', 'changepoint', 'datetime']]
                
                if not data_columns:
                    print(f"  文件 {file} 无有效特征列，跳过")
                    continue
                
                # 处理特征数据
                data = df[data_columns].apply(pd.to_numeric, errors='coerce').values
                data = np.nan_to_num(data, nan=0.0)
                n_samples = len(data)
                if n_samples == 0:
                    print(f"  文件 {file} 无有效样本，跳过")
                    continue
                
                # 生成标签
                if subdir == 'anomaly-free':
                    labels = np.zeros(n_samples, dtype=int)
                    print(f"  加载 anomaly-free 文件 {file}：{n_samples}个正常样本（标签0）")
                else:
                    if 'anomaly' not in df.columns:
                        print(f"  警告：文件 {file} 无'anomaly'列，默认标记为正常样本")
                        labels = np.zeros(n_samples, dtype=int)
                    else:
                        labels_raw = df['anomaly'].apply(pd.to_numeric, errors='coerce').fillna(0)
                        labels = np.where(labels_raw == 1.0, 1, 0).astype(int)
                        n_anomaly = np.sum(labels)
                        n_normal = n_samples - n_anomaly
                        print(f"  文件 {file}：总样本{n_samples}，异常{n_anomaly}，正常{n_normal}")
                
                # 分离样本到正常/异常列表
                if subdir == 'anomaly-free':
                    normal_data.append(data)
                    normal_labels.append(labels)
                else:
                    normal_mask = labels == 0
                    anomaly_mask = labels == 1
                    normal_data.append(data[normal_mask])
                    normal_labels.append(labels[normal_mask])
                    anomaly_data.append(data[anomaly_mask])
                    anomaly_labels.append(labels[anomaly_mask])
        
        # 合并样本
        print("\n===== 合并样本 =====")
        if normal_data:
            normal_data = np.vstack(normal_data)
            normal_labels = np.concatenate(normal_labels)
            print(f"合并后正常样本：{len(normal_labels)}个")
        else:
            normal_data, normal_labels = np.array([]), np.array([])
            print("警告：无正常样本")
        
        if anomaly_data:
            anomaly_data = np.vstack(anomaly_data)
            anomaly_labels = np.concatenate(anomaly_labels)
            print(f"合并后异常样本：{len(anomaly_labels)}个")
        else:
            anomaly_data, anomaly_labels = np.array([]), np.array([])
            print("警告：无异常样本（原始数据应包含异常样本）")
        
        # 验证样本存在性
        if len(anomaly_labels) == 0 and len(normal_labels) > 0:
            print("严重警告：最终数据中无异常样本，可能标签提取错误")
        if len(normal_labels) == 0 and len(anomaly_labels) > 0:
            print("严重警告：最终数据中无正常样本，可能标签提取错误")
        
        # 内部打乱
        np.random.seed(42)
        if len(normal_data) > 0:
            idx = np.random.permutation(len(normal_data))
            normal_data, normal_labels = normal_data[idx], normal_labels[idx]
        if len(anomaly_data) > 0:
            idx = np.random.permutation(len(anomaly_data))
            anomaly_data, anomaly_labels = anomaly_data[idx], anomaly_labels[idx]
        
        # 合并后整体打乱
        if len(normal_data) > 0 and len(anomaly_data) > 0:
            data = np.vstack([normal_data, anomaly_data])
            labels = np.concatenate([normal_labels, anomaly_labels])
            total_idx = np.random.permutation(len(data))
            data, labels = data[total_idx], labels[total_idx]
        elif len(normal_data) > 0:
            data, labels = normal_data, normal_labels
        else:
            data, labels = anomaly_data, anomaly_labels
        
        # 验证整体分布
        total_samples = len(labels)
        total_normal = np.sum(labels == 0)
        total_anomaly = np.sum(labels == 1)
        print(f"\n合并并打乱后总样本：{total_samples}，正常{total_normal}，异常{total_anomaly}，异常比例：{total_anomaly/total_samples:.4f}")
        
        return data, labels
    
    def _split_data(self, full_data: np.ndarray, full_labels: np.ndarray, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """根据split划分数据（复用全量数据，不重复加载）"""
        total_samples = len(full_labels)
        if total_samples == 0:
            raise ValueError("无有效样本，无法划分数据集")
        
        n_train = int(0.7 * total_samples)
        n_val = int(0.8 * total_samples)  # val占10%，test占20%
        
        print(f"\n划分{split}集（总样本{total_samples}）：")
        if split == 'train':
            data_split, labels_split = full_data[:n_train], full_labels[:n_train]
        elif split == 'val':
            data_split, labels_split = full_data[n_train:n_val], full_labels[n_train:n_val]
        elif split == 'test':
            data_split, labels_split = full_data[n_val:], full_labels[n_val:]
        else:
            raise ValueError(f"无效的split：{split}，可选值：train/val/test")
        
        # 验证划分后的数据分布
        split_normal = np.sum(labels_split == 0)
        split_anomaly = np.sum(labels_split == 1)
        print(f"{split}集样本：{len(labels_split)}，正常{split_normal}，异常{split_anomaly}，异常比例：{split_anomaly/len(labels_split):.4f}")
        
        return data_split, labels_split

class SWaTDataset(BaseTimeSeriesDataset):
    """SWaT数据集加载器（适配3个CSV文件：2个正常+1个攻击）"""
    
    # 类变量：存储全量数据（只加载一次）
    _normal_data = None
    _normal_labels = None
    _attack_data = None
    _attack_labels = None
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train', normalize: bool = True):
        # 若全量数据未加载，则加载一次
        if SWaTDataset._normal_data is None or SWaTDataset._attack_data is None:
            SWaTDataset._normal_data, SWaTDataset._normal_labels, \
            SWaTDataset._attack_data, SWaTDataset._attack_labels = self._load_full_swat_data(data_path)
        
        # 从全量数据中划分当前split
        data, labels = self._split_data(split)
        super().__init__(data, labels, window_size, stride, normalize)
    
    def _load_full_swat_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载全量SWaT数据（只执行一次）"""
        # 定义三个文件的预期名称
        normal_files = [
            "SWaT_Dataset_Normal_v0.CSV",
            "SWaT_Dataset_Normal_v1.CSV"
        ]
        attack_file = "SWaT_Dataset_Attack_v0.CSV"
        
        # 检查文件是否存在
        missing_files = []
        for f in normal_files + [attack_file]:
            if not os.path.exists(os.path.join(data_path, f)):
                missing_files.append(f)
        if missing_files:
            raise FileNotFoundError(f"SWaT数据集缺失文件：{missing_files}，请检查路径 {data_path}")
        
        # --------------------------
        # 1. 加载正常数据（用于训练和验证）
        # --------------------------
        normal_data_list = []
        normal_labels_list = []
        for normal_f in normal_files:
            file_path = os.path.join(data_path, normal_f)
            try:
                df = pd.read_csv(file_path, skiprows=1)  # 跳过第二行单位说明
            except Exception as e:
                raise RuntimeError(f"读取正常文件 {normal_f} 失败：{e}")
            
            df.columns = df.columns.str.strip()
            feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Normal/Attack']]
            if not feature_cols:
                raise ValueError(f"正常文件 {normal_f} 无有效特征列，列名：{df.columns.tolist()}")
            
            # 转换特征为数值型，处理异常值
            data = df[feature_cols].apply(pd.to_numeric, errors='coerce').values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 正常文件标签全为0
            labels = np.zeros(len(data), dtype=int)
            
            normal_data_list.append(data)
            normal_labels_list.append(labels)
            print(f"加载正常文件 {normal_f}：样本数 {len(data)}，全为正常样本（标签0）")
        
        # 合并两个正常文件的数据
        normal_data = np.vstack(normal_data_list)
        normal_labels = np.concatenate(normal_labels_list)
        total_normal = len(normal_labels)
        print(f"合并后正常数据总样本：{total_normal}")
        
        # --------------------------
        # 2. 加载攻击数据（用于测试）
        # --------------------------
        attack_path = os.path.join(data_path, attack_file)
        try:
            attack_df = pd.read_csv(attack_path, skiprows=[1])
        except Exception as e:
            raise RuntimeError(f"读取攻击文件 {attack_file} 失败：{e}")
        
        attack_df.columns = attack_df.columns.str.strip()
        if 'Normal/Attack' not in attack_df.columns:
            raise ValueError(
                f"攻击文件 {attack_file} 未找到'Normal/Attack'列，实际列名：\n"
                f"{attack_df.columns.tolist()}"
            )
        attack_feature_cols = [col for col in attack_df.columns if col not in ['Timestamp', 'Normal/Attack']]
        if not attack_feature_cols:
            raise ValueError(f"攻击文件 {attack_file} 无有效特征列")
        
        # 转换攻击数据特征
        attack_data = attack_df[attack_feature_cols].apply(pd.to_numeric, errors='coerce').values
        attack_data = np.nan_to_num(attack_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 攻击文件标签：'Normal'→0，'Attack'→1
        attack_labels = np.where(attack_df['Normal/Attack'] == 'Attack', 1, 0).astype(int)
        print(f"加载攻击文件 {attack_file}：总样本 {len(attack_data)}，异常样本 {np.sum(attack_labels)}，正常样本 {len(attack_data)-np.sum(attack_labels)}")
        
        return normal_data, normal_labels, attack_data, attack_labels
    
    def _split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """根据split划分数据（复用全量数据）"""
        total_normal = len(SWaTDataset._normal_labels)
        
        if split == 'train':
            # 训练集：正常数据的90%
            train_size = int(0.9 * total_normal)
            data = SWaTDataset._normal_data[:train_size]
            labels = SWaTDataset._normal_labels[:train_size]
            print(f"训练集：正常样本 {len(labels)}（占正常数据90%）")
        
        elif split == 'val':
            # 验证集：正常数据的10%
            train_size = int(0.9 * total_normal)
            data = SWaTDataset._normal_data[train_size:]
            labels = SWaTDataset._normal_labels[train_size:]
            print(f"验证集：正常样本 {len(labels)}（占正常数据10%）")
        
        elif split == 'test':
            # 测试集：攻击文件数据
            data = SWaTDataset._attack_data
            labels = SWaTDataset._attack_labels
            print(f"测试集：总样本 {len(labels)}，异常比例 {np.sum(labels)/len(labels):.4f}")
        
        else:
            raise ValueError(f"无效的split：{split}，可选值：train/val/test")
        
        return data, labels
        
class MITBIHDataset(BaseTimeSeriesDataset):
    """MIT-BIH数据集加载器（修复异常心跳识别）"""
    
    # 类变量：存储全量数据
    _full_data = None
    _full_labels = None
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train', normalize: bool = True):
        # 若全量数据未加载，则加载一次
        if MITBIHDataset._full_data is None or MITBIHDataset._full_labels is None:
            MITBIHDataset._full_data, MITBIHDataset._full_labels = self._load_full_mitbih_data(data_path)
        
        # 从全量数据中划分当前split
        data, labels = self._split_data(split)
        super().__init__(data, labels, window_size, stride, normalize)
    
    def _load_full_mitbih_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载全量MIT-BIH数据"""
        all_data = []
        all_labels = []
        
        # 获取所有记录文件（.dat文件）
        record_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
        record_names = list(set([f.split('.')[0] for f in record_files]))
        if not record_names:
            raise FileNotFoundError(f"No MIT-BIH .dat files found in {data_path}")
        
        # 定义异常心跳符号（参考MIT-BIH官方文档）
        ANOMALY_SYMBOLS = {'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'}
        
        # 加载所有记录
        for record_name in record_names:
            try:
                # 读取记录和注释
                record = wfdb.rdrecord(os.path.join(data_path, record_name))
                annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
                
                # 获取ECG信号（取第一通道，单变量）
                ecg_data = record.p_signal[:, 0:1]  # 保持二维结构
                n_samples = len(ecg_data)
                if n_samples == 0:
                    print(f"跳过记录 {record_name}：无有效信号")
                    continue
                
                # 初始化标签（0=正常，1=异常）
                labels = np.zeros(n_samples, dtype=int)
                anomaly_window = 3  # 异常心跳前后3帧标记为异常
                
                # 遍历注释，标记异常区间
                for sample, symbol in zip(annotation.sample, annotation.symbol):
                    if symbol in ANOMALY_SYMBOLS and 0 <= sample < n_samples:
                        start = max(0, sample - anomaly_window)
                        end = min(n_samples - 1, sample + anomaly_window)
                        labels[start:end+1] = 1  # 标记异常区间
                
                # 统计当前记录的异常比例
                n_anomaly = np.sum(labels)
                print(f"加载记录 {record_name}：总样本{n_samples}，异常样本{n_anomaly}，异常比例{n_anomaly/n_samples:.4f}")
                
                all_data.append(ecg_data)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"警告：加载记录 {record_name} 失败，跳过 -> {e}")
                continue
        
        if not all_data:
            raise FileNotFoundError(f"No valid MIT-BIH data found in {data_path}")
        
        # 合并所有序列
        full_data = np.vstack(all_data)
        full_labels = np.concatenate(all_labels)
        print(f"合并后MIT-BIH总样本：{len(full_labels)}")
        
        return full_data, full_labels
    
    def _split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """根据split划分数据（复用全量数据）"""
        n_total = len(MITBIHDataset._full_labels)
        
        if split == 'train':
            data = MITBIHDataset._full_data[:int(0.7 * n_total)]
            labels = MITBIHDataset._full_labels[:int(0.7 * n_total)]
        elif split == 'val':
            data = MITBIHDataset._full_data[int(0.7 * n_total):int(0.8 * n_total)]
            labels = MITBIHDataset._full_labels[int(0.7 * n_total):int(0.8 * n_total)]
        elif split == 'test':
            data = MITBIHDataset._full_data[int(0.8 * n_total):]
            labels = MITBIHDataset._full_labels[int(0.8 * n_total):]
        else:
            raise ValueError(f"无效的split：{split}，可选值：train/val/test")
        
        return data, labels


class DataManager:
    """数据管理器 - 带缓存功能"""
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = config['experiment']['data_root']
        self.cache = {}  # 缓存格式: (dataset_name, split) -> dataset_instance

    def load_dataset(self, dataset_name: str, split: str = 'train') -> BaseTimeSeriesDataset:
        """加载指定数据集，使用缓存避免重复加载"""
        # 生成缓存键（包含所有影响数据集实例化的参数）
        cache_key = (
            dataset_name,
            split,
            self.config['data']['window_size'],
            self.config['data']['stride'],
            self.config['data'].get('normalize', True)
        )
        
        # 检查缓存，存在则直接返回
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 缓存未命中，创建数据集实例
        dataset_config = self.config['datasets'][dataset_name]
        data_path = os.path.join(self.data_root, dataset_config['path'])
        dataset = None
        
        if dataset_name == 'NAB':
            dataset = NABDataset(
                data_path,
                self.config['data']['window_size'],
                self.config['data']['stride'],
                split,
                normalize=self.config['data'].get('normalize', True)
            )
        elif dataset_name == 'SKAB':
            dataset = SKABDataset(
                data_path,
                self.config['data']['window_size'],
                self.config['data']['stride'],
                split,
                normalize=self.config['data'].get('normalize', True)
            )
        elif dataset_name == 'MIT-BIH':
            dataset = MITBIHDataset(
                data_path,
                self.config['data']['window_size'],
                self.config['data']['stride'],
                split,
                normalize=self.config['data'].get('normalize', True)
            )
        elif dataset_name == 'SWaT':
            dataset = SWaTDataset(
                data_path,
                self.config['data']['window_size'],
                self.config['data']['stride'],
                split,
                normalize=self.config['data'].get('normalize', True)
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # 存入缓存
        self.cache[cache_key] = dataset
        return dataset  

    
    def get_data_loader(self, dataset: BaseTimeSeriesDataset, batch_size: int, 
                       shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Windows系统建议设为0，避免多进程错误
            pin_memory=True
        )
    
    def get_normal_samples(self, dataset: BaseTimeSeriesDataset) -> List[torch.Tensor]:
        """获取正常样本（用于对比学习预训练）"""
        normal_samples = []
        for data, label in dataset.samples:
            if label == 0:  # 正常样本
                normal_samples.append(torch.FloatTensor(np.asarray(data, dtype=np.float32)))
        return normal_samples
