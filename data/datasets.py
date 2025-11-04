import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import wfdb  # 用于读取MIT-BIH数据
import json
from typing import Dict, List, Tuple, Optional, Union

class BaseTimeSeriesDataset(Dataset):
    """时间序列数据集基类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 window_size: int = 100, stride: int = 1):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[np.ndarray, int]]:
        """创建滑动窗口样本"""
        samples = []
        n_samples = (len(self.data) - self.window_size) // self.stride + 1
        
        for i in range(n_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]
            window_labels = self.labels[start_idx:end_idx]
            
            # 如果窗口内任何点为异常，则整个窗口标记为异常
            label = 1 if np.any(window_labels == 1) else 0
            samples.append((window_data, label))
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)


class NABDataset(BaseTimeSeriesDataset):
    """NAB数据集加载器"""
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train'):
        data, labels = self._load_nab_data(data_path, split)
        super().__init__(data, labels, window_size, stride)
    
    def _load_nab_data(self, data_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载NAB数据集"""
        # 加载数据文件
        data_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        all_data = []
        all_labels = []
        
        for file in data_files:
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            
            # 假设第一列是数值，第二列是时间戳（如果有）
            values = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
            
            # 加载标签文件
            label_file = file.replace('.csv', '_labels.json')
            label_path = os.path.join(data_path, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_info = json.load(f)
                labels = self._create_labels(len(values), label_info)
            else:
                labels = np.zeros(len(values))
            
            all_data.append(values)
            all_labels.append(labels)
        
        # 合并所有序列
        data = np.concatenate(all_data)
        labels = np.concatenate(all_labels)
        
        # 数据集划分
        n_total = len(data)
        if split == 'train':
            data = data[:int(0.7 * n_total)]
            labels = labels[:int(0.7 * n_total)]
        elif split == 'val':
            data = data[int(0.7 * n_total):int(0.8 * n_total)]
            labels = labels[int(0.7 * n_total):int(0.8 * n_total)]
        elif split == 'test':
            data = data[int(0.8 * n_total):]
            labels = labels[int(0.8 * n_total):]
        
        return data.reshape(-1, 1), labels
    
    def _create_labels(self, length: int, label_info: dict) -> np.ndarray:
        """根据标签信息创建标签数组"""
        labels = np.zeros(length)
        
        if 'anomalies' in label_info:
            for anomaly in label_info['anomalies']:
                start = anomaly.get('start', 0)
                end = anomaly.get('end', length)
                labels[start:end] = 1
                
        return labels


class SKABDataset(BaseTimeSeriesDataset):
    """SKAB数据集加载器"""
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train'):
        data, labels = self._load_skab_data(data_path, split)
        super().__init__(data, labels, window_size, stride)
    
    def _load_skab_data(self, data_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载SKAB数据集"""
        all_data = []
        all_labels = []
        
        # 遍历所有子目录
        for subdir in ['valve1', 'valve2', 'other']:
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.exists(subdir_path):
                continue
                
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            
            for file in csv_files:
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                
                # SKAB数据格式：包含多个传感器列和anomaly列
                data_columns = [col for col in df.columns if col not in ['anomaly', 'changepoint', 'timestamp']]
                labels = df['anomaly'].values if 'anomaly' in df.columns else np.zeros(len(df))
                
                all_data.append(df[data_columns].values)
                all_labels.append(labels)
        
        if not all_data:
            raise FileNotFoundError(f"No valid SKAB data found in {data_path}")
        
        # 合并所有序列
        data = np.vstack(all_data)
        labels = np.concatenate(all_labels)
        
        # 数据集划分
        n_total = len(data)
        if split == 'train':
            data = data[:int(0.7 * n_total)]
            labels = labels[:int(0.7 * n_total)]
        elif split == 'val':
            data = data[int(0.7 * n_total):int(0.8 * n_total)]
            labels = labels[int(0.7 * n_total):int(0.8 * n_total)]
        elif split == 'test':
            data = data[int(0.8 * n_total):]
            labels = labels[int(0.8 * n_total):]
        
        return data, labels


class MITBIHDataset(BaseTimeSeriesDataset):
    """MIT-BIH数据集加载器"""
    
    def __init__(self, data_path: str, window_size: int = 100, 
                 stride: int = 1, split: str = 'train'):
        data, labels = self._load_mitbih_data(data_path, split)
        super().__init__(data, labels, window_size, stride)
    
    def _load_mitbih_data(self, data_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载MIT-BIH数据集"""
        all_data = []
        all_labels = []
        
        # 获取所有记录文件
        record_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
        record_names = list(set([f.split('.')[0] for f in record_files]))
        
        for record_name in record_names[:10]:  # 使用前10个记录作为示例
            try:
                # 读取记录
                record = wfdb.rdrecord(os.path.join(data_path, record_name))
                annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
                
                # 获取ECG信号
                ecg_data = record.p_signal  # 双通道ECG
                
                # 创建标签（将异常心跳标记为异常）
                labels = np.zeros(len(ecg_data))
                for ann_sample in annotation.sample:
                    if ann_sample < len(labels):
                        # 简单示例：将所有标注点视为异常
                        labels[ann_sample] = 1
                
                all_data.append(ecg_data)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Error loading record {record_name}: {e}")
                continue
        
        if not all_data:
            raise FileNotFoundError(f"No valid MIT-BIH data found in {data_path}")
        
        # 合并所有序列
        data = np.vstack(all_data)
        labels = np.concatenate(all_labels)
        
        # 数据集划分
        n_total = len(data)
        if split == 'train':
            data = data[:int(0.7 * n_total)]
            labels = labels[:int(0.7 * n_total)]
        elif split == 'val':
            data = data[int(0.7 * n_total):int(0.8 * n_total)]
            labels = labels[int(0.7 * n_total):int(0.8 * n_total)]
        elif split == 'test':
            data = data[int(0.8 * n_total):]
            labels = labels[int(0.8 * n_total):]
        
        return data, labels


class DataManager:
    """数据管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = config['experiment']['data_root']
        self.scalers = {}
        
    def load_dataset(self, dataset_name: str, split: str = 'train') -> BaseTimeSeriesDataset:
        """加载指定数据集"""
        dataset_config = self.config['datasets'][dataset_name]
        data_path = os.path.join(self.data_root, dataset_config['path'])
        
        if dataset_name == 'NAB':
            return NABDataset(data_path, 
                            self.config['data']['window_size'],
                            self.config['data']['stride'],
                            split)
        elif dataset_name == 'SKAB':
            return SKABDataset(data_path,
                             self.config['data']['window_size'],
                             self.config['data']['stride'],
                             split)
        elif dataset_name == 'MIT-BIH':
            return MITBIHDataset(data_path,
                               self.config['data']['window_size'],
                               self.config['data']['stride'],
                               split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def get_data_loader(self, dataset: BaseTimeSeriesDataset, batch_size: int, 
                       shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(dataset, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=2, pin_memory=True)
    
    def get_normal_samples(self, dataset: BaseTimeSeriesDataset) -> List[torch.Tensor]:
        """获取正常样本（用于对比学习预训练）"""
        normal_samples = []
        for data, label in dataset.samples:
            if label == 0:  # 正常样本
                normal_samples.append(data)
        return normal_samples