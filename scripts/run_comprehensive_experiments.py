"""
综合实验运行脚本 - 覆盖所有数据集和模型组合
SCI要求：系统化的实验设计，完整的配置覆盖
"""
import argparse
import sys
import os
from pathlib import Path
import itertools
import subprocess
import pandas as pd
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class ComprehensiveExperimentRunner:
    """综合实验运行器"""
    
    def __init__(self):
        self.datasets = ['NAB', 'SWaT', 'SKAB', 'MIT-BIH']
        self.encoders = ['TCN', 'LSTM', 'Transformer']
        self.detectors = ['Linear', 'MLP', 'Temporal']
        
        # 实验阶段配置
        self.experiment_phases = {
            'sensitivity': {
                'description': '温度系数敏感性分析',
                'command_template': 'python scripts/run_hyperparameter_tuning.py --task temperature_sensitivity --dataset {dataset} --encoder {encoder}'
            },
            'tuning': {
                'description': '超参数调优',
                'command_template': 'python scripts/run_hyperparameter_tuning.py --task tune_hyperparameters --dataset {dataset} --encoder {encoder} --n_trials 50'
            },
            'baselines': {
                'description': '基线实验',
                'command_template': 'python scripts/run_hyperparameter_tuning.py --task run_baselines --dataset {dataset}'
            },
            'evaluation': {
                'description': '模型评估',
                # 注意：这里移除了 --run_evaluation 参数，因为 main.py 可能需要调整
                'command_template': 'python main.py --dataset {dataset} --encoder {encoder} --detector {detector} --use_wandb'
            }
        }
    
    def generate_experiment_matrix(self) -> pd.DataFrame:
        """生成实验矩阵 - 所有可能的配置组合"""
        experiments = []
        
        for dataset, encoder, detector in itertools.product(self.datasets, self.encoders, self.detectors):
            experiments.append({
                'dataset': dataset,
                'encoder': encoder,
                'detector': detector,
                'experiment_id': f"{dataset}_{encoder}_{detector}",
                'status': 'pending',
                'priority': self._calculate_priority(dataset, encoder, detector)
            })
        
        return pd.DataFrame(experiments)
    
    def _calculate_priority(self, dataset: str, encoder: str, detector: str) -> int:
        """计算实验优先级"""
        priority = 0
        
        # 数据集优先级
        dataset_priority = {'NAB': 4, 'SWaT': 3, 'SKAB': 2, 'MIT-BIH': 1}
        priority += dataset_priority.get(dataset, 0)
        
        # 编码器优先级  
        encoder_priority = {'TCN': 3, 'LSTM': 2, 'Transformer': 1}
        priority += encoder_priority.get(encoder, 0)
        
        # 检测器优先级
        detector_priority = {'MLP': 3, 'Temporal': 2, 'Linear': 1}
        priority += detector_priority.get(detector, 0)
        
        return priority
    
    def run_phase(self, phase: str, specific_config: Dict[str, Any] = None):
        """运行特定实验阶段"""
        print(f"\n🎯 开始阶段: {self.experiment_phases[phase]['description']}")
        print("=" * 60)
        
        if phase == 'baselines':
            # 基线实验按数据集运行
            for dataset in self.datasets:
                print(f"\n📊 在 {dataset} 上运行基线实验...")
                command = self.experiment_phases[phase]['command_template'].format(dataset=dataset)
                self._run_command(command, f"baselines_{dataset}")
        
        elif phase == 'sensitivity' or phase == 'tuning':
            # 敏感性和调优实验按数据集和编码器运行
            for dataset in self.datasets:
                for encoder in self.encoders:
                    print(f"\n🔧 在 {dataset} 上对 {encoder} 进行{self.experiment_phases[phase]['description']}...")
                    command = self.experiment_phases[phase]['command_template'].format(
                        dataset=dataset, encoder=encoder
                    )
                    self._run_command(command, f"{phase}_{dataset}_{encoder}")
        
        elif phase == 'evaluation':
            # 完整评估按所有组合运行
            experiment_matrix = self.generate_experiment_matrix()
            
            # 按优先级排序
            experiment_matrix = experiment_matrix.sort_values('priority', ascending=False)
            
            for _, experiment in experiment_matrix.iterrows():
                print(f"\n🧪 运行实验: {experiment['experiment_id']}")
                command = self.experiment_phases[phase]['command_template'].format(
                    dataset=experiment['dataset'],
                    encoder=experiment['encoder'], 
                    detector=experiment['detector']
                )
                self._run_command(command, experiment['experiment_id'])
    
    def _run_command(self, command: str, experiment_id: str):
        """运行单个命令"""
        print(f"执行: {command}")
        
        try:
            # 创建实验特定日志
            log_dir = Path("experiment_logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{experiment_id}.log"
            
            with open(log_file, 'w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # 实时输出日志
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    print(f"✅ {experiment_id} 完成")
                else:
                    print(f"❌ {experiment_id} 失败，返回码: {process.returncode}")
                    
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")
    
    def generate_experiment_report(self):
        """生成实验进度报告"""
        print("\n📈 实验进度报告")
        print("=" * 60)
        
        # 这里可以添加检查各实验完成状态的逻辑
        # 生成统计报告等
        
        experiment_matrix = self.generate_experiment_matrix()
        print(f"总实验配置: {len(experiment_matrix)}")
        print(f"数据集: {len(self.datasets)}")
        print(f"编码器: {len(self.encoders)}") 
        print(f"检测器: {len(self.detectors)}")
        
        # 保存实验矩阵
        experiment_matrix.to_csv("experiment_matrix.csv", index=False)
        print("✅ 实验矩阵已保存到 experiment_matrix.csv")

def main():
    parser = argparse.ArgumentParser(description='运行综合实验计划')
    parser.add_argument('--phase', type=str, required=True,
                       choices=['sensitivity', 'tuning', 'baselines', 'evaluation', 'all', 'report'],
                       help='要运行的实验阶段')
    parser.add_argument('--dataset', type=str, default='all',
                       help='特定数据集 (默认: all)')
    parser.add_argument('--encoder', type=str, default='all',
                       help='特定编码器 (默认: all)')
    parser.add_argument('--detector', type=str, default='all', 
                       help='特定检测器 (默认: all)')
    
    args = parser.parse_args()
    
    runner = ComprehensiveExperimentRunner()
    
    if args.phase == 'report':
        runner.generate_experiment_report()
        return
    
    print("🔬 综合时间序列异常检测实验")
    print("=" * 60)
    print(f"阶段: {args.phase}")
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder}") 
    print(f"检测器: {args.detector}")
    print("=" * 60)
    
    if args.phase == 'all':
        # 按顺序运行所有阶段
        phases = ['sensitivity', 'tuning', 'baselines', 'evaluation']
        for phase in phases:
            runner.run_phase(phase)
    else:
        runner.run_phase(args.phase)
    
    print("\n🎉 实验阶段完成!")

if __name__ == '__main__':
    main()