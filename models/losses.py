import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class InfoNCELoss(nn.Module):
    """InfoNCE对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        
        # 归一化特征
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 正样本对是对角线元素
        positives = torch.diag(similarity_matrix)
        
        # 负样本对是其他所有元素
        negatives = similarity_matrix
        
        # 计算对比损失
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(negatives), dim=1)
        
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class TemporalContrastiveLoss(nn.Module):
    """时间对比损失"""
    
    def __init__(self, margin: float = 0.5, time_gap: int = 10):
        super().__init__()
        self.margin = margin
        self.time_gap = time_gap
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 相邻时间步相似度损失
        adjacent_loss = 0
        for t in range(seq_len - 1):
            sim = F.cosine_similarity(hidden_states[:, t], hidden_states[:, t+1], dim=-1)
            adjacent_loss += (1 - sim).mean()
        adjacent_loss /= (seq_len - 1)
        
        # 远距离时间步差异损失
        distant_loss = 0
        count = 0
        for t in range(seq_len):
            for k in range(self.time_gap, min(seq_len - t, self.time_gap * 2)):
                sim = F.cosine_similarity(hidden_states[:, t], hidden_states[:, t+k], dim=-1)
                distant_loss += F.relu(self.margin - sim).mean()
                count += 1
        distant_loss = distant_loss / count if count > 0 else torch.tensor(0.0)
        
        return adjacent_loss + distant_loss


class FrequencyContrastiveLoss(nn.Module):
    """频率对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, time_domain_features: torch.Tensor, 
                freq_domain_features: torch.Tensor) -> torch.Tensor:
        # 归一化特征
        time_features = F.normalize(time_domain_features, dim=1)
        freq_features = F.normalize(freq_domain_features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(time_features, freq_features.T) / self.temperature
        
        # 正样本对是对角线元素
        positives = torch.diag(similarity_matrix)
        
        # 负样本对是其他所有元素
        negatives = similarity_matrix
        
        # 计算对比损失
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(negatives), dim=1)
        
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class CombinedContrastiveLoss(nn.Moderule):
    """组合对比损失"""
    
    def __init__(self, temp_nce: float = 0.07, margin_temp: float = 0.5,
                 temp_freq: float = 0.07, weights: List[float] = None):
        super().__init__()
        self.nce_loss = InfoNCELoss(temp_nce)
        self.temp_loss = TemporalContrastiveLoss(margin_temp)
        self.freq_loss = FrequencyContrastiveLoss(temp_freq)
        
        self.weights = weights or [1.0, 0.5, 0.5]
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, 
                hidden_states: torch.Tensor = None,
                freq_features: torch.Tensor = None) -> torch.Tensor:
        
        loss = self.weights[0] * self.nce_loss(z1, z2)
        
        if hidden_states is not None:
            loss += self.weights[1] * self.temp_loss(hidden_states)
            
        if freq_features is not None:
            loss += self.weights[2] * self.freq_loss(z1, freq_features)
            
        return loss