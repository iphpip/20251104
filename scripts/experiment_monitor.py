"""
实验监控器 - 实时跟踪实验进度和资源使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import time
from datetime import datetime

class ExperimentMonitor:
    def __init__(self):
        self.experiment_matrix = pd.read_csv("experiment_matrix.csv")
        self.results_dir = Path("results")
        
    def get_experiment_status(self):
        """获取实验状态概览"""
        status_summary = {
            'total_experiments': len(self.experiment_matrix),
            'completed': 0,
            'running': 0,
            'failed': 0,
            'pending': len(self.experiment_matrix)
        }
        
        # 检查各实验状态（简化实现）
        for exp_id in self.experiment_matrix['experiment_id']:
            result_file = self.results_dir / f"{exp_id}_results.json"
            if result_file.exists():
                status_summary['completed'] += 1
                status_summary['pending'] -= 1
        
        return status_summary
    
    def generate_progress_report(self):
        """生成进度报告"""
        status = self.get_experiment_status()
        
        print("\n📊 实验进度报告")
        print("=" * 50)
        print(f"总实验数: {status['total_experiments']}")
        print(f"已完成: {status['completed']} ({status['completed']/status['total_experiments']*100:.1f}%)")
        print(f"进行中: {status['running']}")
        print(f"失败: {status['failed']}") 
        print(f"待进行: {status['pending']}")
        print("=" * 50)
        
        # 预计完成时间
        if status['completed'] > 0:
            avg_time_per_exp = 2.5  # 小时，根据实际情况调整
            remaining_time = (status['pending'] * avg_time_per_exp) / 8  # 假设并行8个实验
            print(f"预计剩余时间: {remaining_time:.1f} 天")
    
    def monitor_resource_usage(self):
        """监控资源使用情况"""
        # 这里可以添加GPU内存、CPU使用率等监控
        print("🔍 资源监控功能待实现")

if __name__ == '__main__':
    monitor = ExperimentMonitor()
    monitor.generate_progress_report()