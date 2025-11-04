"""
超参数调优运行脚本
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tuning.hyperparameter_tuner import HyperparameterTuner, run_temperature_sensitivity_analysis
from baselines.baseline_experiments import BaselineExperiments

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning and baseline experiments')
    parser.add_argument('--task', type=str, required=True,
                       choices=['tune_hyperparameters', 'temperature_sensitivity', 'run_baselines', 'all'])
    parser.add_argument('--dataset', type=str, default='NAB',
                       choices=['NAB', 'SWaT', 'SKAB', 'MIT-BIH', 'all'])
    parser.add_argument('--encoder', type=str, default='TCN',
                       choices=['TCN', 'LSTM', 'Transformer'])
    parser.add_argument('--detector', type=str, default='MLP',
                       choices=['Linear', 'MLP', 'Temporal'])
    parser.add_argument('--n_trials', type=int, default=50)
    
    args = parser.parse_args()
    
    print("SCIENTIFIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    if args.task in ['tune_hyperparameters', 'all']:
        # 超参数调优
        study_name = f"{args.dataset}_{args.encoder}_{args.detector}_optimization"
        tuner = HyperparameterTuner(
            config_path='configs/default.yaml',
            study_name=study_name,
            n_trials=args.n_trials
        )
        
        best_params = tuner.run_study()
        print(f"Best parameters: {best_params}")
    
    if args.task in ['temperature_sensitivity', 'all']:
        # 温度敏感性分析
        if args.dataset == 'all':
            datasets = ['NAB', 'SWaT', 'SKAB', 'MIT-BIH']
        else:
            datasets = [args.dataset]
            
        if args.encoder == 'all':
            encoders = ['TCN', 'LSTM', 'Transformer']
        else:
            encoders = [args.encoder]
            
        for dataset in datasets:
            for encoder in encoders:
                run_temperature_sensitivity_analysis('configs/default.yaml', dataset, encoder)
    
    if args.task in ['run_baselines', 'all']:
        # 基线实验
        baseline_runner = BaselineExperiments('configs/default.yaml')
        
        if args.dataset == 'all':
            datasets = ['NAB', 'SWaT', 'SKAB', 'MIT-BIH']
        else:
            datasets = [args.dataset]
        
        all_results = {}
        for dataset in datasets:
            print(f"\n Running baselines on {dataset}")
            results = baseline_runner.run_all_baselines(dataset)
            all_results[dataset] = results
        
        # 比较分析
        from baselines.baseline_experiments import BaselineComparison
        comparator = BaselineComparison()
        report = comparator.create_comparison_report(all_results)
        
        print("Baseline experiments completed and report generated")

if __name__ == '__main__':
    main()
