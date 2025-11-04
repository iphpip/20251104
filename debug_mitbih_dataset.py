import os
import numpy as np
import yaml
from data.datasets import MITBIHDataset  # 假设已实现MIT-BIH数据集类

def check_mitbih_ratio(config):
    """检查MIT-BIH数据集的异常比例配置合理性（心电异常检测）"""
    dataset_name = "MIT-BIH"
    dataset_config = config['datasets'][dataset_name]
    data_root = config['experiment']['data_root']
    data_path = os.path.join(data_root, dataset_config['path'])
    window_size = config['data']['window_size']
    stride = config['data']['stride']
    
    print(f"===== 开始检查 {dataset_name} 数据集 =====")
    print(f"数据集路径：{data_path}")
    print(f"配置的训练集异常比例：{dataset_config['train_anomaly_ratio']:.4f}")
    print(f"配置的测试集异常比例：{dataset_config['test_anomaly_ratio']:.4f}\n")

    try:
        # 加载训练集（标签：1=异常心跳，0=正常心跳）
        train_set = MITBIHDataset(
            data_path=data_path,
            window_size=window_size,
            stride=stride,
            split='train',
            normalize=False
        )
        train_labels = train_set.labels  # 假设已转换为二值标签（1=异常）
        train_total = len(train_labels)
        train_anomaly = np.sum(train_labels == 1)
        train_ratio = train_anomaly / train_total if train_total > 0 else 0.0

        # 加载测试集
        test_set = MITBIHDataset(
            data_path=data_path,
            window_size=window_size,
            stride=stride,
            split='test',
            normalize=False
        )
        test_labels = test_set.labels
        test_total = len(test_labels)
        test_anomaly = np.sum(test_labels == 1)
        test_ratio = test_anomaly / test_total if test_total > 0 else 0.0

        # 输出实际统计
        print(f"【训练集实际统计】总样本：{train_total}，异常样本：{train_anomaly}，实际异常比例：{train_ratio:.4f}")
        print(f"【测试集实际统计】总样本：{test_total}，异常样本：{test_anomaly}，实际异常比例：{test_ratio:.4f}\n")

        # 对比配置与实际（允许±0.005误差，因心电异常比例本身较低）
        train_diff = abs(train_ratio - dataset_config['train_anomaly_ratio'])
        test_diff = abs(test_ratio - dataset_config['test_anomaly_ratio'])
        print(f"配置与实际差异：训练集 {train_diff:.4f}，测试集 {test_diff:.4f}")

        if train_diff <= 0.005 and test_diff <= 0.005:
            print(f"✅ {dataset_name} 配置合理（差异在允许范围内）")
        else:
            print(f"❌ {dataset_name} 配置不合理！建议调整：")
            print(f"train_anomaly_ratio: {train_ratio:.4f}")
            print(f"test_anomaly_ratio: {test_ratio:.4f}")

    except Exception as e:
        print(f"❌ 加载 {dataset_name} 失败：{str(e)}")

if __name__ == "__main__":
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    check_mitbih_ratio(config)
    print("\n===== 检查结束 =====")