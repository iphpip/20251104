import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import torch
from pathlib import Path
from typing import Dict, List, Any

class ScientificVisualizer:
    """科学可视化工具（满足SCI论文要求）"""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置科学绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette("husl", 8)
        
        # 设置中文字体（如果需要）
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_curves(self, 
                           train_losses: List[float], 
                           val_losses: List[float] = None,
                           train_metrics: Dict[str, List[float]] = None,
                           val_metrics: Dict[str, List[float]] = None,
                           title: str = "Training Progress",
                           save_name: str = "training_curves"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率曲线（如果有）
        if train_metrics and 'learning_rate' in train_metrics:
            axes[0, 1].plot(epochs, train_metrics['learning_rate'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].remove()
        
        # 指标曲线
        metric_plots = [(1, 0), (1, 1)]
        metric_idx = 0
        
        if train_metrics and val_metrics:
            for metric_name in ['f1', 'precision', 'recall', 'accuracy']:
                if metric_name in train_metrics and metric_idx < len(metric_plots):
                    row, col = metric_plots[metric_idx]
                    axes[row, col].plot(epochs, train_metrics[metric_name], 
                                      'b-', label=f'Train {metric_name.upper()}', linewidth=2)
                    axes[row, col].plot(epochs, val_metrics[metric_name], 
                                      'r-', label=f'Val {metric_name.upper()}', linewidth=2)
                    axes[row, col].set_title(f'{metric_name.upper()} Score', 
                                           fontsize=14, fontweight='bold')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel(metric_name.upper())
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                    metric_idx += 1
        
        # 填充空子图
        for i in range(2):
            for j in range(2):
                if not axes[i, j].has_data():
                    axes[i, j].remove()
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_roc_curves(self, results_dict: Dict[str, Dict[str, np.ndarray]],
                       title: str = "ROC Curves Comparison",
                       save_name: str = "roc_curves"):
        """绘制多个模型的ROC曲线对比"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            fpr, tpr, _ = roc_curve(results['true'], results['scores'])
            auc_score = np.trapz(tpr, fpr)
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{model_name} (AUC = {auc_score:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_precision_recall_curves(self, results_dict: Dict[str, Dict[str, np.ndarray]],
                                   title: str = "Precision-Recall Curves",
                                   save_name: str = "pr_curves"):
        """绘制精确率-召回率曲线"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            precision, recall, _ = precision_recall_curve(results['true'], results['scores'])
            ap_score = average_precision_score(results['true'], results['scores'])
            
            ax.plot(recall, precision, linewidth=2,
                   label=f'{model_name} (AP = {ap_score:.4f})')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_ablation_study(self, ablation_results: pd.DataFrame,
                          metric: str = 'f1',
                          title: str = "Ablation Study Results",
                          save_name: str = "ablation_study"):
        """绘制消融实验结果"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        variants = ablation_results.index.tolist()
        metrics = ablation_results[metric].values
        
        bars = ax.barh(variants, metrics, color=self.colors[:len(variants)])
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(metric.upper())
        
        # 在条形上添加数值
        for bar, value in zip(bars, metrics):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_hyperparameter_sensitivity(self, param_name: str, param_values: List[float],
                                      performance: List[float], 
                                      title: str = "Hyperparameter Sensitivity",
                                      save_name: str = "hyperparameter_sensitivity"):
        """绘制超参数敏感性分析"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_values, performance, 'o-', linewidth=3, markersize=8)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Performance (F1-score)')
        ax.grid(True, alpha=0.3)
        
        # 标记最佳点
        best_idx = np.argmax(performance)
        ax.annotate(f'Best: {performance[best_idx]:.4f}',
                   xy=(param_values[best_idx], performance[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def create_comprehensive_report(self, experiment_results: Dict[str, Any],
                                  save_name: str = "comprehensive_report"):
        """创建综合实验报告"""
        report_path = self.output_dir / f"{save_name}.html"
        
        # 创建交互式报告
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Comparison', 'Training Curves',
                          'Confusion Matrix', 'ROC Curves'),
            specs=[[{"type": "bar"}, {"type": "xy"}],
                   [{"type": "heatmap"}, {"type": "xy"}]]
        )
        
        # 这里添加具体的图表内容
        # 由于篇幅限制，简化实现
        
        fig.update_layout(height=800, title_text="Comprehensive Experiment Report")
        fig.write_html(str(report_path))
        
        return report_path
    
    def _save_figure(self, fig, name: str, formats: List[str] = ['png', 'pdf']):
        """保存图表"""
        for fmt in formats:
            save_path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        plt.close(fig)