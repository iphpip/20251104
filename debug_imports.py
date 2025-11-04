"""
调试导入问题
"""
import sys
from pathlib import Path

def debug_import(module_name, class_name=None):
    """调试单个导入"""
    try:
        if class_name:
            exec(f"from {module_name} import {class_name}")
            print(f"✅ {module_name}.{class_name} 导入成功")
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name} 导入成功")
        return True
    except Exception as e:
        print(f"❌ {module_name}{'.' + class_name if class_name else ''} 导入失败: {e}")
        return False

def main():
    print("🔍 调试导入问题")
    print("=" * 50)
    
    # 添加项目根目录
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    # 测试关键导入
    imports_to_test = [
        ("utils.logger", "ScientificLogger"),
        ("data.datasets", "DataManager"),
        ("models.encoders", "TCNEncoder"),
        ("models.projection_heads", "MLPProjectionHead"),
        ("models.detectors", "MLPAnomalyDetector"),
        ("models.losses", "CombinedContrastiveLoss"),
        ("training.pretrainer", "EnhancedContrastivePretrainer"),
        ("training.detector_trainer", "EnhancedAnomalyDetectorTrainer"),
        ("training.evaluator", "ScientificEvaluator"),
        ("data.augmentation", "ContrastiveAugmentor"),
        ("tuning.hyperparameter_tuner", "HyperparameterTuner"),
        ("baselines.baseline_experiments", "BaselineExperiments"),
    ]
    
    success_count = 0
    for module, cls in imports_to_test:
        if debug_import(module, cls):
            success_count += 1
    
    print("=" * 50)
    print(f"导入成功率: {success_count}/{len(imports_to_test)}")
    
    if success_count == len(imports_to_test):
        print("🎉 所有导入都成功！")
    else:
        print("❌ 有些导入失败，请检查上述错误信息")

if __name__ == '__main__':
    main()