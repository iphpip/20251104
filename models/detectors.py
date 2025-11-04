import torch
import torch.nn as nn
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KernelDensity
from typing import List

class AnomalyScorer:
    """异常评分器"""
    
    def __init__(self, method: str = 'cosine'):
        self.method = method
        self.normal_mean = None
        self.normal_cov = None
        self.kde = None
        
    def fit(self, normal_features: np.ndarray):
        """在正常样本上拟合"""
        self.normal_mean = np.mean(normal_features, axis=0)
        
        if self.method == 'mahalanobis':
            self.normal_cov = EmpiricalCovariance().fit(normal_features)
        elif self.method == 'kde':
            self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(normal_features)
            
    def score(self, features: np.ndarray) -> np.ndarray:
        """计算异常分数"""
        if self.method == 'cosine':
            # 余弦相似度评分
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            normal_mean_norm = self.normal_mean / np.linalg.norm(self.normal_mean)
            similarities = np.dot(features_norm, normal_mean_norm)
            return 1 - similarities
            
        elif self.method == 'mahalanobis':
            # 马氏距离评分
            return self.normal_cov.mahalanobis(features - self.normal_mean)
            
        elif self.method == 'kde':
            # 核密度估计评分
            return -self.kde.score_samples(features)
            
        else:
            raise ValueError(f"Unsupported scoring method: {self.method}")


class LinearAnomalyDetector(nn.Module):
    """线性异常检测器"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class MLPAnomalyDetector(nn.Module):
    """MLP异常检测器"""
    
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class TemporalAnomalyDetector(nn.Module):
    """时序异常检测器"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv1d(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)