import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from utils.logger import ScientificLogger

# 之前是class AnomalyDetectorTrainer:
class EnhancedAnomalyDetectorTrainer:
    """异常检测器训练器"""
    
    def __init__(self, encoder: nn.Module, detector: nn.Module, 
                 device: torch.device, config: dict):
        self.encoder = encoder
        self.detector = detector
        self.device = device
        self.config = config
        self.experiment_name = experiment_name
        self.logger = ScientificLogger(experiment_name)
        
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(
            self.detector.parameters(),
            lr=config['learning_rate']
        )
        
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.encoder.eval()
        self.detector.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc='Training Detector')
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.float().to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.encoder(data)
                
            # 异常检测
            scores = self.detector(features)
            
            # 计算损失
            loss = self.criterion(scores, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> dict:
        """验证"""
        self.encoder.eval()
        self.detector.eval()
        
        all_scores = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.float().to(self.device)
                
                # 提取特征
                features = self.encoder(data)
                
                # 异常检测
                scores = self.detector(features)
                
                # 计算损失
                loss = self.criterion(scores, labels)
                total_loss += loss.item()
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # 计算指标
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 找到最佳阈值
        best_threshold, best_f1 = self.find_optimal_threshold(all_scores, all_labels)
        predictions = (all_scores > best_threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, predictions, average='binary', zero_division=0
        )
        
        auc = roc_auc_score(all_labels, all_scores)
        
        return {
            'loss': total_loss / len(dataloader),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': best_threshold
        }
    
    def find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray) -> tuple:
        """寻找最优阈值"""
        best_threshold = 0
        best_f1 = 0
        
        for threshold in np.linspace(0, 1, 100):
            predictions = (scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold, best_f1
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练过程"""
        best_f1 = 0
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val F1 = {val_metrics['f1']:.4f}, "
                  f"Val AUC = {val_metrics['auc']:.4f}")
                  
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
                
            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, val_metrics)
                
    def save_checkpoint(self, epoch: int, metrics: dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'detector_state_dict': self.detector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, f'checkpoints/detector_epoch_{epoch}.pth')