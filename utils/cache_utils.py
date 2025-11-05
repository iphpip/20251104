import shutil
from pathlib import Path

def clear_dataset_cache(cache_dir: str = "cache/datasets"):
    """清理数据集样本缓存"""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        print(f"已清理数据集缓存: {cache_dir}")

def clear_feature_cache(cache_dir: str = "cache/baseline_features"):
    """清理特征缓存"""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        print(f"已清理特征缓存: {cache_dir}")

# 使用示例（在配置变更或代码更新后调用）
if __name__ == "__main__":
    # 当修改了窗口生成逻辑时，清理数据集缓存
    clear_dataset_cache()