import yaml
import torch
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def setup_device(config: Dict[str, Any]) -> torch.device:
    """设置设备"""
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def setup_random_seed(config: Dict[str, Any]):
    """设置随机种子"""
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")

def get_model_config(config: Dict[str, Any], model_type: str, model_name: str) -> Dict[str, Any]:
    """获取模型配置"""
    if model_type == 'encoder':
        return config['encoders'][model_name]
    elif model_type == 'detector':
        return config['detectors'][model_name]
    else:
        raise ValueError(f"Unknown model type: {model_type}")