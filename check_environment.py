"""
环境验证脚本 - 检查所有必要的库是否正确安装
"""
import importlib
import sys
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否可用"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - 成功导入")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - 导入失败: {e}")
        return False

def main():
    print("🔍 检查实验环境依赖...")
    print("=" * 50)
    
    # 必需的核心包
    core_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("optuna", "optuna"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    
    # 重要的可选包
    optional_packages = [
        ("tsfresh", "tsfresh"),
        ("statsmodels", "statsmodels"),
        ("wfdb", "wfdb"),
        ("pyod", "pyod"),
        ("wandb", "wandb"),
        ("scikit-posthocs", "scikit_posthocs"),
    ]
    
    print("核心依赖包检查:")
    print("-" * 30)
    core_success = 0
    for pkg_name, import_name in core_packages:
        if check_package(pkg_name, import_name):
            core_success += 1
    
    print("\n可选依赖包检查:")
    print("-" * 30)
    optional_success = 0
    for pkg_name, import_name in optional_packages:
        if check_package(pkg_name, import_name):
            optional_success += 1
    
    print("\n" + "=" * 50)
    print(f"环境检查完成:")
    print(f"核心包: {core_success}/{len(core_packages)} 个成功")
    print(f"可选包: {optional_success}/{len(optional_packages)} 个成功")
    
    if core_success == len(core_packages):
        print("🎉 所有核心依赖包安装成功！可以开始实验。")
        
        # 检查PyTorch GPU支持
        import torch
        if torch.cuda.is_available():
            print(f"🎯 GPU可用: {torch.cuda.get_device_name()}")
            print(f"🎯 CUDA版本: {torch.version.cuda}")
        else:
            print("⚠️  GPU不可用，将使用CPU运行（训练速度会较慢）")
            
    else:
        print("❌ 部分核心依赖包安装失败，请检查安装。")

if __name__ == "__main__":
    main()