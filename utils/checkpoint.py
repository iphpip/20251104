import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建检查点索引
        self.index_file = self.checkpoint_dir / "checkpoint_index.json"
        self.checkpoint_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """加载检查点索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": [], "best_models": {}}
    
    def _save_index(self):
        """保存检查点索引"""
        with open(self.index_file, 'w') as f:
            json.dump(self.checkpoint_index, f, indent=2)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       model_type: str = "encoder",
                       experiment_name: str = "default") -> str:
        """保存检查点"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{experiment_name}_{model_type}_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 保存模型状态
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'model_type': model_type,
            'experiment_name': experiment_name
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 更新索引
        checkpoint_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'model_type': model_type,
            'experiment_name': experiment_name
        }
        
        self.checkpoint_index["checkpoints"].append(checkpoint_info)
        
        # 保存最佳模型
        if is_best:
            best_key = f"{experiment_name}_{model_type}"
            self.checkpoint_index["best_models"][best_key] = checkpoint_info
            
            # 创建最佳模型的符号链接（便于快速访问）
            best_path = self.checkpoint_dir / f"BEST_{best_key}.pth"
            if best_path.exists():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path.name)
        
        self._save_index()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        return torch.load(checkpoint_path, map_location='cpu')
    
    def get_best_checkpoint(self, experiment_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """获取最佳模型检查点"""
        best_key = f"{experiment_name}_{model_type}"
        if best_key in self.checkpoint_index["best_models"]:
            checkpoint_info = self.checkpoint_index["best_models"][best_key]
            return self.load_checkpoint(checkpoint_info['path'])
        return None
    
    def get_latest_checkpoint(self, experiment_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """获取最新检查点"""
        relevant_checkpoints = [
            cp for cp in self.checkpoint_index["checkpoints"]
            if cp['experiment_name'] == experiment_name and cp['model_type'] == model_type
        ]
        
        if not relevant_checkpoints:
            return None
        
        latest = max(relevant_checkpoints, key=lambda x: x['timestamp'])
        return self.load_checkpoint(latest['path'])
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """清理旧检查点，只保留最近的n个"""
        for model_type in set(cp['model_type'] for cp in self.checkpoint_index["checkpoints"]):
            for experiment_name in set(cp['experiment_name'] for cp in self.checkpoint_index["checkpoints"]):
                relevant_checkpoints = [
                    cp for cp in self.checkpoint_index["checkpoints"]
                    if cp['experiment_name'] == experiment_name and cp['model_type'] == model_type
                ]
                
                if len(relevant_checkpoints) > keep_last_n:
                    # 按时间排序，删除旧的
                    relevant_checkpoints.sort(key=lambda x: x['timestamp'])
                    checkpoints_to_remove = relevant_checkpoints[:-keep_last_n]
                    
                    for cp in checkpoints_to_remove:
                        cp_path = Path(cp['path'])
                        if cp_path.exists():
                            cp_path.unlink()
                        self.checkpoint_index["checkpoints"].remove(cp)
        
        self._save_index()