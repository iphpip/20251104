#!/usr/bin/env python3
"""
科学实验运行脚本 - 确保SCI级别的实验严谨性
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run scientific experiments for SCI publication')
    parser.add_argument('--phase', type=str, required=True,
                       choices=['pretrain', 'detector', 'evaluate', 'ablation', 'full'],
                       help='Experiment phase to run')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['NAB', 'SWaT', 'SKAB', 'MIT-BIH', 'all'])
    parser.add_argument('--rigor_level', type=str, default='high',
                       choices=['standard', 'high', 'very_high'],
                       help='Level of statistical rigor')
    
    args = parser.parse_args()
    
    print("🧪 SCIENTIFIC EXPERIMENT FRAMEWORK")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Dataset: {args.dataset}")
    print(f"Rigor Level: {args.rigor_level}")
    print("=" * 60)
    
    # 根据严谨级别设置参数
    if args.rigor_level == 'very_high':
        n_runs = 10  # 高严谨性：10次运行
        confidence_level = 0.99
    elif args.rigor_level == 'high':
        n_runs = 5   # 中等严谨性：5次运行  
    else:
        n_runs = 3   # 标准严谨性：3次运行
    
    print(f"Number of runs: {n_runs}")
    print(f"Confidence level: {confidence_level}")
    
    # 这里调用主实验逻辑
    run_scientific_experiment(args.phase, args.dataset, n_runs, confidence_level)

if __name__ == '__main__':
    main()