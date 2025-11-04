"""
基于对比学习的时间序列异常检测实验框架
主要功能：
1. 对比学习预训练 - 在正常样本上学习通用表示
2. 异常检测器训练 - 冻结编码器，训练检测头
3. 全面评估 - 多维度性能评估和统计检验
4. 消融实验 - 分析各组件贡献度
SCI要求：完整的实验流程，严格的评估标准，统计显著性检验
"""
import torch
import argparse
import wandb
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config import load_config, setup_device, setup_random_seed, get_model_config
from utils.logger import ExperimentLogger
from data.datasets import DataManager
from data.augmentation import ContrastiveAugmentor
from models.encoders import TCNEncoder, LSTMEncoder, TransformerEncoder
from models.projection_heads import MLPProjectionHead
from models.detectors import LinearAnomalyDetector, MLPAnomalyDetector, TemporalAnomalyDetector, ContrastiveModel
from models.losses import CombinedContrastiveLoss
from training.pretrainer import EnhancedContrastivePretrainer
from training.detector_trainer import EnhancedAnomalyDetectorTrainer
from training.evaluator import ComprehensiveEvaluator
from analysis.experiment_analyzer import ExperimentAnalyzer

def create_experiment_name(dataset: str, encoder: str, detector: str) -> str:
    """创建实验名称"""
    timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset}_{encoder}_{detector}_{timestamp}"

def main():
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection with Contrastive Learning')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset', type=str, default='NAB', 
                       choices=['NAB', 'SKAB', 'MIT-BIH'])
    parser.add_argument('--encoder', type=str, default='TCN',
                       choices=['TCN', 'LSTM', 'Transformer'])
    parser.add_argument('--detector', type=str, default='MLP',
                       choices=['Linear', 'MLP', 'Temporal'])
    parser.add_argument('--run_pretrain', action='store_true', help='Run contrastive pre-training')
    parser.add_argument('--run_detector', action='store_true', help='Run anomaly detector training')
    parser.add_argument('--run_evaluation', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    config['model']['encoder_type'] = args.encoder
    config['model']['detector_type'] = args.detector
    
    # 创建实验名称
    if not args.experiment_name:
        experiment_name = create_experiment_name(args.dataset, args.encoder, args.detector)
    else:
        experiment_name = args.experiment_name
    
    # 设置环境
    device = setup_device(config)
    setup_random_seed(config)
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=config['experiment']['name'],
            name=experiment_name,
            config=config
        )
    
    # 创建目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    encoder = None
    detector = None
    
    try:
        # 预训练阶段
        if args.run_pretrain:
            encoder = run_pretraining(config, device, args.dataset, experiment_name)
        
        # 检测器训练阶段
        if args.run_detector:
            if encoder is None:
                encoder = load_pretrained_encoder(config, device, args.dataset, args.encoder, experiment_name)
            if encoder is not None:
                detector = run_detector_training(config, device, args.dataset, experiment_name, encoder)
        
        # 评估阶段
        if args.run_evaluation:
            if encoder is None:
                encoder = load_pretrained_encoder(config, device, args.dataset, args.encoder, experiment_name)
            if detector is None:
                detector = load_trained_detector(config, device, args.detector, experiment_name)
            if encoder is not None and detector is not None:
                results = run_evaluation(config, device, args.dataset, experiment_name, encoder, detector)
        
        # 消融实验
        if args.run_ablation:
            run_ablation_studies(config, device, args.dataset, experiment_name)
        
    except Exception as e:
        print(f"Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()