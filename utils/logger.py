"""
科学实验日志系统
功能：
1. 详细记录实验配置、训练过程、评估结果
2. 自动保存实验记录为JSON格式
3. 实时监控训练指标和模型状态
4. 生成实验时间线和检查点记录
SCI要求：完整的实验记录，便于审稿人验证实验过程
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from typing import Dict, Any
import numpy as np

class ScientificLogger:
    """科学实验日志记录器 - 满足SCI论文严谨性要求"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 科学实验记录结构
        self.experiment_record = {
            "experiment_metadata": {
                "name": experiment_name,
                "start_time": datetime.now().isoformat(),
                "purpose": "Contrastive Learning for Time Series Anomaly Detection",
                "research_questions": [
                    "Does contrastive pre-training improve anomaly detection performance?",
                    "Which encoder architecture works best for time series representation?",
                    "How does the method perform across different anomaly types?"
                ]
            },
            "methodology": {
                "config": {},
                "model_architecture": {},
                "training_procedure": {}
            },
            "results": {
                "pretraining": [],
                "detector_training": [],
                "evaluation": {},
                "ablation_studies": {},
                "statistical_tests": {}
            },
            "artifacts": {
                "checkpoints": [],
                "visualizations": [],
                "metrics_files": []
            }
        }
        
        self._setup_logging()
        self._log_experiment_start()
    
    def _setup_logging(self):
        """设置科学实验级别的日志记录"""
        self.logger = logging.getLogger(f"sci_experiment_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复handler
        if not self.logger.handlers:
            # 详细文件日志
            file_handler = logging.FileHandler(
                self.log_dir / "detailed_experiment.log", 
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 控制台输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 科学论文格式
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_research_design(self, research_design: Dict[str, Any]):
        """记录研究设计 - SCI论文方法部分"""
        self.experiment_record["methodology"].update(research_design)
        self.logger.info("Research design documented")
        self._save_record()
    
    def log_hyperparameter_search(self, search_results: Dict[str, Any]):
        """记录超参数搜索过程和结果"""
        if "hyperparameter_optimization" not in self.experiment_record["methodology"]:
            self.experiment_record["methodology"]["hyperparameter_optimization"] = []
        
        search_results["timestamp"] = datetime.now().isoformat()
        self.experiment_record["methodology"]["hyperparameter_optimization"].append(search_results)
        
        self.logger.info(f"Hyperparameter search completed: {search_results['method']}")
        self._save_record()
    
    def log_statistical_test(self, test_name: str, results: Dict[str, Any], 
                           interpretation: str = ""):
        """记录统计检验结果 - SCI论文结果部分"""
        statistical_test = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "interpretation": interpretation,
            "significance_level": 0.05
        }
        
        self.experiment_record["results"]["statistical_tests"][test_name] = statistical_test
        self.logger.info(f"Statistical test {test_name}: p-value = {results.get('p_value', 'N/A')}")
        self._save_record()
    
    def log_ablation_result(self, ablation_config: str, results: Dict[str, float], 
                          relative_performance: Dict[str, float]):
        """记录消融实验结果 - SCI论文分析部分"""
        ablation_record = {
            "configuration": ablation_config,
            "timestamp": datetime.now().isoformat(),
            "absolute_performance": results,
            "relative_performance": relative_performance,
            "performance_change": self._calculate_performance_change(relative_performance)
        }
        
        if "ablation_studies" not in self.experiment_record["results"]:
            self.experiment_record["results"]["ablation_studies"] = {}
        
        self.experiment_record["results"]["ablation_studies"][ablation_config] = ablation_record
        self.logger.info(f"Ablation study: {ablation_config} completed")
        self._save_record()
    
    def _calculate_performance_change(self, relative_perf: Dict[str, float]) -> Dict[str, str]:
        """计算性能变化程度"""
        changes = {}
        for metric, change in relative_perf.items():
            if change > 0.1:
                changes[metric] = "Large improvement"
            elif change > 0.05:
                changes[metric] = "Moderate improvement" 
            elif change > -0.05:
                changes[metric] = "Negligible change"
            elif change > -0.1:
                changes[metric] = "Moderate degradation"
            else:
                changes[metric] = "Large degradation"
        return changes
    
    def _log_experiment_start(self):
        """记录实验开始信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"SCIENTIFIC EXPERIMENT: {self.experiment_name}")
        self.logger.info(f"Start Time: {self.experiment_record['experiment_metadata']['start_time']}")
        self.logger.info("Research Questions:")
        for i, question in enumerate(self.experiment_record["experiment_metadata"]["research_questions"], 1):
            self.logger.info(f"  {i}. {question}")
        self.logger.info("=" * 80)
    
    def _save_record(self):
        """保存实验记录"""
        record_file = self.log_dir / "scientific_experiment_record.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_record, f, indent=2, ensure_ascii=False)
    
    def finalize_experiment(self, conclusions: Dict[str, Any]):
        """完成实验并记录结论"""
        self.experiment_record["experiment_metadata"]["end_time"] = datetime.now().isoformat()
        self.experiment_record["conclusions"] = conclusions
        
        duration = datetime.fromisoformat(
            self.experiment_record["experiment_metadata"]["end_time"]
        ) - datetime.fromisoformat(
            self.experiment_record["experiment_metadata"]["start_time"]
        )
        
        self.experiment_record["experiment_metadata"]["duration_hours"] = duration.total_seconds() / 3600
        
        self._save_record()
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT COMPLETED")
        self.logger.info(f"Duration: {duration}")
        self.logger.info("Conclusions:")
        for key, value in conclusions.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)