import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           precision_recall_curve, roc_curve)
from scipy.stats import wilcoxon
from typing import Dict, Tuple

class TimeSeriesMetrics:
    """时间序列异常检测评估指标"""
    
    @staticmethod
    def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_scores: np.ndarray) -> Dict:
        """计算基础分类指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_scores),
            'auc_pr': average_precision_score(y_true, y_scores)
        }
    
    @staticmethod
    def compute_temporal_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               anomaly_ranges: list) -> Dict:
        """计算时序特定指标"""
        # 序列级准确率
        sequence_accuracy = TimeSeriesMetrics.sequence_level_accuracy(
            y_true, y_pred, anomaly_ranges
        )
        
        # 定位精度
        localization_precision = TimeSeriesMetrics.anomaly_localization(
            y_true, y_pred, anomaly_ranges
        )
        
        # 延迟检测时间
        detection_delay = TimeSeriesMetrics.detection_delay(
            y_true, y_pred, anomaly_ranges
        )
        
        return {
            'sequence_accuracy': sequence_accuracy,
            'localization_precision': localization_precision,
            'detection_delay': detection_delay
        }
    
    @staticmethod
    def sequence_level_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                              anomaly_ranges: list) -> float:
        """序列级准确率"""
        correct_detections = 0
        
        for start, end in anomaly_ranges:
            anomaly_segment = y_true[start:end]
            pred_segment = y_pred[start:end]
            
            if np.any(anomaly_segment == 1) and np.any(pred_segment == 1):
                correct_detections += 1
                
        return correct_detections / len(anomaly_ranges) if anomaly_ranges else 0
    
    @staticmethod
    def anomaly_localization(y_true: np.ndarray, y_pred: np.ndarray,
                           anomaly_ranges: list) -> float:
        """异常段定位精度"""
        total_overlap = 0
        total_anomaly_length = 0
        
        for start, end in anomaly_ranges:
            anomaly_segment = y_true[start:end]
            pred_segment = y_pred[start:end]
            
            overlap = np.sum((anomaly_segment == 1) & (pred_segment == 1))
            total_overlap += overlap
            total_anomaly_length += np.sum(anomaly_segment == 1)
            
        return total_overlap / total_anomaly_length if total_anomaly_length > 0 else 0
    
    @staticmethod
    def detection_delay(y_true: np.ndarray, y_pred: np.ndarray,
                       anomaly_ranges: list) -> float:
        """延迟检测时间"""
        delays = []
        
        for start, end in anomaly_ranges:
            anomaly_times = np.where(y_true[start:end] == 1)[0]
            pred_times = np.where(y_pred[start:end] == 1)[0]
            
            if len(pred_times) > 0 and len(anomaly_times) > 0:
                first_anomaly = anomaly_times[0]
                first_detection = pred_times[0]
                delay = max(0, first_detection - first_anomaly)
                delays.append(delay)
                
        return np.mean(delays) if delays else 0
    
    @staticmethod
    def statistical_significance_test(results_a: list, results_b: list, 
                                    metric: str = 'f1') -> Dict:
        """统计显著性检验"""
        stat, p_value = wilcoxon(results_a, results_b)
        
        # 计算效应量
        cohens_d = np.abs(np.mean(results_a) - np.mean(results_b)) / np.std(
            np.concatenate([results_a, results_b])
        )
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }