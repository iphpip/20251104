"""
基线实验模块 - 实现和比较传统异常检测方法
SCI要求：全面的基线比较，公平的实验设置
"""
import numpy as np
import pandas as pd
import json  # 修复1：新增json导入（用于保存报告）
import matplotlib.pyplot as plt  # 修复2：新增matplotlib导入（用于可视化）
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
# 修复3：替换sklearn的LOF为pyod的LOF（支持测试集预测，避免数据泄露）
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from utils.config import load_config, setup_device
from utils.logger import ScientificLogger
from utils.metrics import TimeSeriesMetrics
from data.datasets import DataManager

class BaselineExperiments:
    """基线实验运行器"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = setup_device(self.config)
        self.metrics_calculator = TimeSeriesMetrics()
        
        # 创建基线结果目录
        self.baseline_dir = Path("baseline_results")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = ScientificLogger("baseline_experiments")
        
        # 记录基线方法描述
        self.logger.log_research_design({
            "baseline_methods": {
                "Statistical Methods": ["3Sigma", "HampelFilter", "EllipticEnvelope"],
                "Traditional ML": ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
                "Deep Learning": ["AutoEncoder", "LSTMAutoEncoder"],
                "State-of-the-Art": ["AnomalyTransformer"]
            },
            "evaluation_criteria": "Same dataset splits, same evaluation metrics, same hardware"
        })
    
    def run_all_baselines(self, dataset: str) -> pd.DataFrame:
        """运行所有基线方法"""
        print(f"Running Baseline Experiments on {dataset}")
        
        # 加载数据（仅加载1次，避免重复加载）
        data_manager = DataManager(self.config)
        train_dataset = data_manager.load_dataset(dataset, 'train')
        test_dataset = data_manager.load_dataset(dataset, 'test')
        
        # 准备数据（仅处理1次，生成统计特征）
        X_train, y_train = self.prepare_data(train_dataset)
        X_test, y_test = self.prepare_data(test_dataset)
        
        baseline_results = {}
        
        # 1. 统计方法
        print("1. Statistical Methods...")
        baseline_results.update(self.run_statistical_methods(X_train, X_test, y_test))
        
        # 2. 传统机器学习方法
        print("2. Traditional ML Methods...")
        baseline_results.update(self.run_traditional_ml_methods(X_train, X_test, y_test))
        
        # 3. 深度学习方法
        print("3. Deep Learning Methods...")
        baseline_results.update(self.run_deep_learning_methods(X_train, X_test, y_test, train_dataset))
        
        # 转换为DataFrame
        results_df = pd.DataFrame(baseline_results).T
        
        # 保存结果到CSV
        results_df.to_csv(self.baseline_dir / f"{dataset}_baseline_results.csv")
        
        # 修复4：替换不存在的log_analysis，使用新增的log_evaluation_result记录结果
        self.logger.log_evaluation_result("baseline_comparison", {
            "dataset": dataset,
            "results": results_df.to_dict(),
            "best_method": results_df['f1'].idxmax(),
            "performance_summary": results_df.describe().to_dict()
        })
        
        return results_df
    
    def run_statistical_methods(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """运行统计基线方法"""
        results = {}
        
        # 1. 3-Sigma原则
        print("  - 3-Sigma Rule")
        scores_3sigma = self.three_sigma_rule(X_train, X_test)
        results['3Sigma'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_3sigma > 0.5, scores_3sigma)
        
        # 2. Hampel滤波器
        print("  - Hampel Filter")
        scores_hampel = self.hampel_filter(X_train, X_test)
        results['HampelFilter'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_hampel > 0.5, scores_hampel)
        
        # 3. 椭圆包络
        print("  - Elliptic Envelope")
        try:
            envelope = EllipticEnvelope(contamination=0.1, random_state=42)
            envelope.fit(X_train)
            scores_envelope = -envelope.decision_function(X_test)  # 负号：分数越高越异常
            results['EllipticEnvelope'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_envelope > 0, scores_envelope)
        except Exception as e:
            print(f"EllipticEnvelope failed: {e}")
            results['EllipticEnvelope'] = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        
        return results
    
    def run_traditional_ml_methods(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """运行传统机器学习方法"""
        results = {}
        
        # 1. 隔离森林
        print("  - Isolation Forest")
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        iso_forest.fit(X_train)
        scores_iso = -iso_forest.decision_function(X_test)  # 负号：分数越高越异常
        results['IsolationForest'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_iso > 0, scores_iso)
        
        # 2. 一类SVM
        print("  - One-Class SVM")
        try:
            oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
            oc_svm.fit(X_train)
            scores_svm = -oc_svm.decision_function(X_test)  # 负号：分数越高越异常
            results['OneClassSVM'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_svm > 0, scores_svm)
        except Exception as e:
            print(f"OneClassSVM failed: {e}")
            results['OneClassSVM'] = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        
        # 修复5：用pyod的LOF替换sklearn的LOF（支持测试集预测，避免重新fit测试集）
        print("  - Local Outlier Factor")
        try:
            lof = LOF(contamination=0.1, n_neighbors=20, novelty=True)  # novelty=True：支持新样本预测
            lof.fit(X_train)
            test_scores_lof = lof.decision_function(X_test)  # 直接对测试集计算异常分数
            results['LocalOutlierFactor'] = self.metrics_calculator.compute_basic_metrics(y_test, test_scores_lof > 0, test_scores_lof)
        except Exception as e:
            print(f"LocalOutlierFactor failed: {e}")
            results['LocalOutlierFactor'] = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        
        return results
    
    def run_deep_learning_methods(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, train_dataset) -> Dict[str, Dict]:
        """运行深度学习基线方法"""
        results = {}
        
        # 1. 自动编码器（pyod的AutoEncoder，支持epochs参数）
        print("  - AutoEncoder")
        try:
            ae = AutoEncoder(epochs=50, contamination=0.1, random_state=42, hidden_neurons=[64, 32, 32, 64])
            ae.fit(X_train)
            scores_ae = ae.decision_function(X_test)  # 异常分数：越高越异常
            results['AutoEncoder'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_ae > 0.5, scores_ae)
        except Exception as e:
            print(f"AutoEncoder failed: {e}")
            results['AutoEncoder'] = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        
        # 2. LSTM自动编码器（当前为演示代码，需替换为实际实现）
        print("  - LSTM AutoEncoder")
        try:
            scores_lstm_ae = self.lstm_autoencoder(train_dataset, X_test, y_test)
            results['LSTMAutoEncoder'] = self.metrics_calculator.compute_basic_metrics(y_test, scores_lstm_ae > 0.5, scores_lstm_ae)
        except Exception as e:
            print(f"LSTM AutoEncoder failed: {e}")
            results['LSTMAutoEncoder'] = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        
        return results
    
    def prepare_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """准备基线方法的数据（提取时间序列统计特征）"""
        features = []
        labels = []
        
        for data, label in dataset:
            data = np.asarray(data, dtype=np.float32)
            stats_features = []
            
            # 对每个维度提取统计特征（均值、标准差等）
            for dim in range(data.shape[1] if len(data.shape) > 1 else 1):
                dim_data = data[:, dim] if len(data.shape) > 1 else data
                # 统一用numpy计算（避免torch和numpy混用）
                stats_features.extend([
                    np.mean(dim_data),
                    np.std(dim_data),
                    np.min(dim_data),
                    np.max(dim_data),
                    np.median(dim_data),
                    np.percentile(dim_data, 25),
                    np.percentile(dim_data, 75)
                ])
            features.append(stats_features)            
            labels.append(label if isinstance(label, int) else label.item())  # 确保标签为int
        
        # 标准化特征（避免尺度影响）
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return np.array(features), np.array(labels)
    
    def three_sigma_rule(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """3-Sigma原则异常检测（基于马氏距离简化版）"""
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8  # 避免除以0
        
        # 计算每个测试样本的异常分数（越远越异常）
        anomalies = np.sum((X_test - mean) ** 2 / std ** 2, axis=1)
        return anomalies / np.max(anomalies)  # 归一化到[0,1]
    
    def hampel_filter(self, X_train: np.ndarray, X_test: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Hampel滤波器异常检测（基于中位数绝对偏差）"""
        scores = []
        # 计算训练集的中位数和MAD（用于测试集判断）
        train_median = np.median(X_train, axis=0)
        train_mad = np.median(np.abs(X_train - train_median), axis=0) + 1e-8
        
        for sample in X_test:
            # 异常分数：样本与训练集中位数的偏差 / MAD
            score = np.sum(np.abs(sample - train_median) / (1.4826 * train_mad))  # 1.4826是正态分布校正系数
            scores.append(score)
        
        scores = np.array(scores)
        return scores / np.max(scores)  # 归一化到[0,1]
    
    def lstm_autoencoder(self, train_dataset, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """LSTM自动编码器（演示代码：需替换为实际PyTorch实现）"""
        # 提示：实际实现需定义LSTM-AE网络、训练逻辑、计算重构误差作为异常分数
        # 当前返回随机分数仅用于测试流程，需替换！
        print("  ⚠️ LSTM AutoEncoder为演示代码，返回随机分数，请替换为实际实现！")
        return np.random.random(len(X_test))


class BaselineComparison:
    """基线方法比较分析（修复可视化和未定义类问题）"""
    
    def __init__(self):
        # 修复6：删除未定义的ScientificVisualizer，直接使用matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False  # 支持负号
    
    def create_comparison_report(self, results_dict: Dict[str, pd.DataFrame]):
        """创建基线方法比较报告（修复JSON保存和可视化）"""
        
        # 合并所有数据集的结果
        comparison_data = []
        for dataset, results in results_dict.items():
            for method, metrics in results.iterrows():
                comparison_data.append({
                    'dataset': dataset,
                    'method': method,
                    **metrics.to_dict()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 生成性能比较图（修复可视化代码）
        self.plot_baseline_comparison(comparison_df)
        
        # 统计显著性检验（示例：可补充实际检验逻辑）
        statistical_tests = self.perform_statistical_tests(comparison_df)
        
        # 生成报告字典
        report = {
            'comparison_data': comparison_df.to_dict('records'),
            'statistical_tests': statistical_tests,
            'best_methods_by_dataset': self.identify_best_methods(comparison_df),
            'overall_ranking': self.rank_methods(comparison_df)
        }
        
        # 保存报告到JSON（修复7：确保json已导入）
        with open('baseline_results/comprehensive_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 基线比较报告已保存到 baseline_results/comprehensive_comparison_report.json")
        return report
    
    def plot_baseline_comparison(self, comparison_df: pd.DataFrame):
        """绘制基线方法F1-Score比较图（基础matplotlib实现）"""
        if comparison_df.empty:
            print("⚠️ 无比较数据，跳过可视化")
            return
        
        # 按方法分组计算平均F1-Score和标准差
        method_perf = comparison_df.groupby('method').agg({
            'f1': ['mean', 'std'],
            'auc_roc': 'mean'
        }).round(4)
        method_perf.columns = ['f1_mean', 'f1_std', 'auc_roc_mean']  # 重命名列
        method_perf = method_perf.sort_values('f1_mean', ascending=False)  # 按F1降序排序
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(method_perf))
        bars = ax.bar(
            x_pos, method_perf['f1_mean'], 
            yerr=method_perf['f1_std'], capsize=5, 
            alpha=0.7, color='skyblue'
        )
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10
            )
        
        # 设置图表属性
        ax.set_xlabel('基线方法', fontsize=12)
        ax.set_ylabel('平均F1-Score', fontsize=12)
        ax.set_title('各基线方法在不同数据集上的F1-Score比较', fontsize=14, pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_perf.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)  # F1-Score范围[0,1]
        ax.grid(axis='y', alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('baseline_results/baseline_comparison_f1.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 基线比较图已保存到 baseline_results/baseline_comparison_f1.png")
    
    def perform_statistical_tests(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """统计显著性检验（示例：可补充t检验/ANOVA逻辑）"""
        # 提示：实际SCI论文需补充真实检验（如ANOVA+事后检验），此处为示例结构
        if len(comparison_df['method'].unique()) < 2:
            return {"error": "方法数量不足，无法进行统计检验"}
        
        # 示例：比较最优方法和次优方法的F1-Score差异
        best_method = comparison_df.loc[comparison_df['f1'].idxmax(), 'method']
        second_best_method = comparison_df[comparison_df['method'] != best_method].loc[
            comparison_df[comparison_df['method'] != best_method]['f1'].idxmax(), 'method'
        ]
        
        return {
            "test_name": "示例：最优方法vs次优方法F1差异",
            "best_method": best_method,
            "second_best_method": second_best_method,
            "best_mean_f1": comparison_df[comparison_df['method'] == best_method]['f1'].mean(),
            "second_best_mean_f1": comparison_df[comparison_df['method'] == second_best_method]['f1'].mean(),
            "difference": comparison_df[comparison_df['method'] == best_method]['f1'].mean() - 
                          comparison_df[comparison_df['method'] == second_best_method]['f1'].mean(),
            "note": "实际需补充t检验/ANOVA的p值和显著性判断"
        }
    
    def identify_best_methods(self, comparison_df: pd.DataFrame) -> Dict[str, str]:
        """按数据集识别最优方法（F1-Score最高）"""
        best_methods = {}
        for dataset in comparison_df['dataset'].unique():
            dataset_data = comparison_df[comparison_df['dataset'] == dataset]
            best_method = dataset_data.loc[dataset_data['f1'].idxmax(), 'method']
            best_f1 = dataset_data['f1'].max()
            best_methods[dataset] = f"{best_method} (F1: {best_f1:.3f})"
        return best_methods
    
    def rank_methods(self, comparison_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """按平均F1-Score对方法整体排名"""
        method_ranking = comparison_df.groupby('method').agg({
            'f1': ['mean', 'std'],
            'auc_roc': 'mean'
        }).round(4)
        method_ranking.columns = ['mean_f1', 'std_f1', 'mean_auc_roc']
        method_ranking = method_ranking.sort_values('mean_f1', ascending=False).reset_index()
        method_ranking['rank'] = range(1, len(method_ranking) + 1)
        
        return method_ranking[['rank', 'method', 'mean_f1', 'std_f1', 'mean_auc_roc']].to_dict('records')