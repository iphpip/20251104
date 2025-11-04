"""
简化版主程序 - 用于测试导入问题
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    print("✅ 简化版主程序 - 导入测试")
    
    # 测试导入各个模块
    try:
        from utils.logger import ScientificLogger
        print("✅ ScientificLogger 导入成功")
        
        from data.datasets import DataManager
        print("✅ DataManager 导入成功")
        
        from models.encoders import TCNEncoder
        print("✅ TCNEncoder 导入成功")
        
        from models.projection_heads import MLPProjectionHead
        print("✅ MLPProjectionHead 导入成功")
        
        from models.detectors import MLPAnomalyDetector
        print("✅ MLPAnomalyDetector 导入成功")
        
        from models.losses import CombinedContrastiveLoss
        print("✅ CombinedContrastiveLoss 导入成功")
        
        from training.pretrainer import EnhancedContrastivePretrainer
        print("✅ EnhancedContrastivePretrainer 导入成功")
        
        from training.detector_trainer import EnhancedAnomalyDetectorTrainer
        print("✅ EnhancedAnomalyDetectorTrainer 导入成功")
        
        from training.evaluator import ScientificEvaluator
        print("✅ ScientificEvaluator 导入成功")
        
        from data.augmentation import ContrastiveAugmentor
        print("✅ ContrastiveAugmentor 导入成功")
        
        print("\n🎉 所有模块导入成功！")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())