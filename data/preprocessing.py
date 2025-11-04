import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        
    def z_score_normalize(self, data: np.ndarray, feature_wise: bool = True) -> np.ndarray:
        """Z-score标准化"""
        if feature_wise and data.ndim > 1:
            # 多变量数据，每个特征独立标准化
            normalized_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                if i not in self.scalers:
                    self.scalers[i] = StandardScaler()
                normalized_data[:, i] = self.scalers[i].fit_transform(
                    data[:, i].reshape(-1, 1)).flatten()
            return normalized_data
        else:
            # 单变量数据
            if 'univariate' not in self.scalers:
                self.scalers['univariate'] = StandardScaler()
            return self.scalers['univariate'].fit_transform(data.reshape(-1, 1)).flatten()
    
    def handle_missing_values(self, data: np.ndarray, strategy: str = 'linear') -> np.ndarray:
        """处理缺失值"""
        if strategy == 'linear':
            df = pd.DataFrame(data)
            return df.interpolate(method='linear').values
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            return imputer.fit_transform(data)
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}")
    
    def sliding_window(self, data: np.ndarray, window_size: int, 
                      stride: int = 1) -> np.ndarray:
        """滑动窗口分割"""
        n_windows = (len(data) - window_size) // stride + 1
        windows = np.zeros((n_windows, window_size) + data.shape[1:])
        
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            windows[i] = data[start:end]
            
        return windows
    
    def train_test_split(self, data: np.ndarray, labels: np.ndarray, 
                        train_ratio: float = 0.7, val_ratio: float = 0.1) -> Tuple:
        """数据集划分"""
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        indices = np.random.permutation(n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return (data[train_idx], data[val_idx], data[test_idx],
                labels[train_idx], labels[val_idx], labels[test_idx])