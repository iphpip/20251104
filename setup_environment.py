"""
自动环境设置脚本
"""
import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并检查结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("🚀 开始设置时间序列异常检测实验环境")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return
    
    print(f"✅ Python版本: {sys.version}")
    
    # 创建requirements.txt
    requirements_content = """torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
tsfresh>=0.20.0
statsmodels>=0.14.0
wfdb>=4.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
kaleido>=0.2.1
optuna>=3.2.0
joblib>=1.2.0
pyod>=1.0.0
wandb>=0.15.0
tensorboard>=2.13.0
PyYAML>=6.0
tqdm>=4.65.0
scikit-posthocs>=0.7.0
colorama>=0.4.6
pillow>=9.5.0"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    print("✅ 已创建requirements.txt文件")
    
    # 安装依赖
    if run_command("pip install -r requirements.txt", "安装所有依赖"):
        print("\n🎉 环境设置完成！")
        print("运行 'python check_environment.py' 验证安装")
    else:
        print("\n❌ 安装过程中出现错误，请手动检查")

if __name__ == "__main__":
    main()