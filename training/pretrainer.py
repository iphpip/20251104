import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Any, List

from utils.logger import ScientificLogger
from utils.checkpoint import CheckpointManager
from utils.visualization import ScientificVisualizer

class EnhancedContrastivePretrainer:
    """增强的对比学习预训练器（包含完整日志和检查点）"""
    
    def __init__(self, model: nn.Module, augmentor, loss_fn: nn.Module,
                 device: torch.device, config: dict, experiment_name: str):
        self.model = model
        self.augmentor = augmentor
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.experiment_name = experiment_name
        
        # 初始化工具
        self.logger = ScientificLogger(experiment_name)
        self.checkpoint_manager = CheckpointManager()
        self.visualizer = ScientificVisualizer()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # 记录配置
        self.logger.log_config({
            'experiment_name': experiment_name,
            'model_type': 'contrastive_pretraining',
            'training_config': config,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(device)
        })
        
        self.logger.get_logger().info(f"Initialized pretrainer with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Training Epoch {self.current_epoch}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            batch_size = data.size(0)
            
            try:
                # 生成增强视图
                view1, view2 = [], []
                for i in range(batch_size):
                    v1, v2 = self.augmentor(data[i].cpu().numpy())
                    view1.append(torch.tensor(v1, dtype=torch.float32))
                    view2.append(torch.tensor(v2, dtype=torch.float32))
                    
                view1 = torch.stack(view1).to(self.device)
                view2 = torch.stack(view2).to(self.device)
                
                # 前向传播
                z1 = self.model(view1)
                z2 = self.model(view2)
                
                # 计算损失
                loss = self.loss_fn(z1, z2)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_losses.append(batch_loss)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # 记录批次信息
                if batch_idx % 100 == 0:
                    self.logger.get_logger().debug(
                        f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                        f"Loss = {batch_loss:.4f}, LR = {self.scheduler.get_last_lr()[0]:.6f}"
                    )
                    
                    if wandb.run is not None:
                        wandb.log({
                            'batch_loss': batch_loss,
                            'learning_rate': self.scheduler.get_last_lr()[0]
                        })
                        
            except Exception as e:
                self.logger.get_logger().error(
                    f"Error in batch {batch_idx}: {str(e)}"
                )
                continue
                
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc='Validating'):
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # 生成增强视图
                view1, view2 = [], []
                for i in range(batch_size):
                    v1, v2 = self.augmentor(data[i].cpu().numpy())
                    view1.append(torch.tensor(v1, dtype=torch.float32))
                    view2.append(torch.tensor(v2, dtype=torch.float32))
                    
                view1 = torch.stack(view1).to(self.device)
                view2 = torch.stack(view2).to(self.device)
                
                # 前向传播
                z1 = self.model(view1)
                z2 = self.model(view2)
                
                # 计算损失
                loss = self.loss_fn(z1, z2)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """完整训练过程"""
        self.logger.get_logger().info("Starting contrastive pre-training")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader) if val_loader else None
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录指标
            metrics = {
                'train_loss': train_loss,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
            if val_loss is not None:
                metrics['val_loss'] = val_loss
            
            self.logger.log_metrics('pretrain', metrics, epoch)
            
            # 保存检查点
            is_best = val_loss is not None and val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % self.config['save_interval'] == 0 or is_best:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=is_best,
                    model_type='contrastive_encoder',
                    experiment_name=self.experiment_name
                )
                
                self.logger.log_checkpoint({
                    'name': f'epoch_{epoch}',
                    'path': checkpoint_path,
                    'epoch': epoch,
                    'metrics': metrics,
                    'is_best': is_best
                })
            
            # 记录到wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    **metrics
                })
            
            self.logger.get_logger().info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"{f'Val Loss = {val_loss:.4f}, ' if val_loss else ''}"
                f"Best = {self.best_loss:.4f}"
            )
        
        # 训练完成，生成可视化
        self._generate_training_plots()
        
        self.logger.get_logger().info("Contrastive pre-training completed")
    
    def _generate_training_plots(self):
        """生成训练过程可视化"""
        # 训练曲线
        train_metrics = {
            'learning_rate': self.learning_rates
        }
        
        val_metrics = {} if not self.val_losses else {
            'loss': self.val_losses
        }
        
        self.visualizer.plot_training_curves(
            train_losses=self.train_losses,
            val_losses=self.val_losses if self.val_losses else None,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            title=f"Contrastive Pre-training - {self.experiment_name}",
            save_name=f"{self.experiment_name}_pretraining_curves"
        )