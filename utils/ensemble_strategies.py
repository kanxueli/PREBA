#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多次预测聚合策略评估模块
实现并评估各种多次预测聚合策略在手术时长预测任务上的性能
"""

import json
import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_metrics import calculate_metrics, print_metrics_summary

class EnsembleStrategies:
    """多次预测聚合策略评估器"""
    
    def __init__(self, results_file: str = None, train_df: pd.DataFrame = None, test_df: pd.DataFrame = None, results: List[Dict] = None):
        """
        初始化评估器
        
        Args:
            results_file: 预测结果JSON文件路径（可选，如果提供了results则不需要）
            train_df: 训练集DataFrame（用于动态贝叶斯平均）
            test_df: 测试集DataFrame（用于动态贝叶斯平均）
            results: 预测结果列表（可选，如果提供了则直接使用，不再从文件加载）
        """
        self.results_file = results_file
        if results is not None:
            # 如果直接提供了results，使用它
            self.results = results
        else:
            # 否则从文件加载
            self.results = self._load_results()
        self.strategies = {}
        self.train_df = train_df
        self.test_df = test_df
        self.training_stats = None
        
        # 如果有训练集，构建训练集统计信息
        if self.train_df is not None:
            self._build_training_stats()
        
    def _load_results(self) -> List[Dict]:
        """加载预测结果"""
        if self.results_file is None:
            return []
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"成功加载预测结果: {len(results)} 个样本")
            return results
        except Exception as e:
            print(f"加载结果文件失败: {e}")
            return []
    
    def _get_valid_predictions(self, result: Dict) -> List[float]:
        """获取有效的预测值"""
        all_predictions = result.get('all_predictions', [])
        # 过滤掉None和无效值
        valid_predictions = [p for p in all_predictions if p is not None and p > 0]
        return valid_predictions
    
    def _build_training_stats(self):
        """构建训练集统计信息（按分层匹配策略）"""
        if self.train_df is None:
            return
        
        print("构建训练集统计信息...")
        
        # 定义分层匹配策略
        group_keys_levels = self._get_group_keys_levels()
        
        # 过滤有效数据
        valid_data = self.train_df[
            (self.train_df['手术间时长'] > 0) & 
            (self.train_df['手术间时长'].notna())
        ].copy()
        
        stats = {}
        
        # 按每个分层策略构建统计信息
        for level, keys in enumerate(group_keys_levels):
            # 检查所有必需的列是否存在
            if all(key in valid_data.columns for key in keys):
                # 按这些键分组
                grouped = valid_data.groupby(keys)
                
                for key_tuple, group in grouped:
                    if len(group) >= 50: 
                        mean_duration = group['手术间时长'].mean()
                        median_duration = group['手术间时长'].median()
                        count = len(group)
                        
                        # 构建分层键（与prior_hint_generator.py保持一致）
                        if not isinstance(key_tuple, tuple):
                            key_tuple = (key_tuple,)
                        hierarchical_key = (len(keys),) + key_tuple
                        
                        stats[hierarchical_key] = {
                            'mean': mean_duration,
                            'median': median_duration,
                            'count': count,
                            'keys': keys
                        }
        
        self.training_stats = stats
        print(f"训练集统计信息构建完成，共{len(stats)}个分层组合")
    
    def _get_group_keys_levels(self):
        group_keys_levels = [
            # 4维组合（最高优先级）
            ['患者住院科室','拟行手术名称','手术器械清点单手术名称','手术等级'],
            ['患者住院科室','拟行手术名称','手术器械清点单手术名称','总体评估时的ASA分级'],
            # 3维组合
            ['患者住院科室','拟行手术名称','手术器械清点单手术名称'],
            ['患者住院科室','拟行手术名称','手术等级'],
            ['患者住院科室','手术器械清点单手术名称','手术等级'],
            ['拟行手术名称','手术器械清点单手术名称','手术等级'],
            ['患者住院科室','拟行手术名称','总体评估时的ASA分级'],
            ['患者住院科室','手术等级','总体评估时的ASA分级'],
            # 2维组合
            # ['患者住院科室','拟行手术名称'],
            # ['患者住院科室','手术器械清点单手术名称'],
            # # ['患者住院科室','手术等级'],
            # ['拟行手术名称','手术器械清点单手术名称'],
            # ['拟行手术名称','手术等级'],
            # ['手术器械清点单手术名称','手术等级'],
            # # # 1维组合（最低优先级）
            # ['患者住院科室'],
            # ['拟行手术名称'],
            # ['手术等级']
        ]
        return group_keys_levels

    def _get_prior_info_from_training_stats(self, surgery_id: str) -> Dict:
        """根据surgery_id获取对应的先验信息（分层匹配策略）"""
        if self.training_stats is None or self.test_df is None:
            return None
        
        try:
            # 从测试集中找到对应的记录
            test_record = self.test_df[
                self.test_df['手麻系统中手术状态时间历史记录唯一标识'] == surgery_id
            ]
            
            if test_record.empty:
                return None
            
            # 获取测试样本的特征值
            v = lambda k: test_record.iloc[0].get(k, '')
            
            # 定义分层匹配策略
            group_keys_levels = self._get_group_keys_levels()
            
            # 按优先级查找匹配的统计信息
            for keys in group_keys_levels:
                # 构建当前层级的键值
                key_values = tuple(v(k) for k in keys)
                hierarchical_key = (len(keys),) + key_values
                
                # 查找匹配的统计信息
                if hierarchical_key in self.training_stats:
                    return self.training_stats[hierarchical_key]
            
            return None
                    
        except Exception as e:
            print(f"获取先验信息时出错: {e}")
            return None
    
    def strategy_simple_average(self, predictions: List[float]) -> float:
        """策略1: 简单平均"""
        return np.mean(predictions) if predictions else None
    
    def strategy_trimmed_mean(self, predictions: List[float], trim_ratio: float = 0.2) -> float:
        """策略2: 截尾平均（去掉最高和最低的20%）"""
        if len(predictions) < 3:
            return np.mean(predictions) if predictions else None
        
        sorted_preds = sorted(predictions)
        n_trim = max(1, int(len(predictions) * trim_ratio))
        trimmed_preds = sorted_preds[n_trim:-n_trim] if n_trim > 0 else sorted_preds
        return np.mean(trimmed_preds) if trimmed_preds else None
    
    def strategy_median(self, predictions: List[float]) -> float:
        """策略3: 中位数"""
        return np.median(predictions) if predictions else None
    
    def strategy_weighted_average(self, predictions: List[float], weights: List[float] = None) -> float:
        """策略4: 加权平均（基于预测顺序，越靠前权重越高）"""
        if not predictions:
            return None
        
        if weights is None:
            # 默认权重：第一个预测权重最高，依次递减
            weights = [1.0 / (i + 1) for i in range(len(predictions))]
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return np.average(predictions, weights=weights)
    
    def strategy_robust_mean(self, predictions: List[float]) -> float:
        """策略5: 鲁棒平均（基于MAD的异常值检测）"""
        if len(predictions) < 3:
            return np.mean(predictions) if predictions else None
        
        predictions = np.array(predictions)
        median = np.median(predictions)
        mad = np.median(np.abs(predictions - median))
        
        # 使用MAD进行异常值检测
        threshold = 2.5 * mad
        robust_predictions = predictions[np.abs(predictions - median) <= threshold]
        
        return np.mean(robust_predictions) if len(robust_predictions) > 0 else median
    
    def strategy_confidence_weighted(self, predictions: List[float], temperatures: List[float] = None) -> float:
        """策略6: 基于温度的置信度加权"""
        if not predictions:
            return None
        
        if temperatures is None or len(temperatures) != len(predictions):
            return np.mean(predictions)
        
        # 温度越低，置信度越高
        confidences = [1.0 / (temp + 0.1) for temp in temperatures]
        confidences = np.array(confidences)
        confidences = confidences / np.sum(confidences)
        
        return np.average(predictions, weights=confidences)
    
    def strategy_quantile_average(self, predictions: List[float], quantiles: List[float] = [0.25, 0.5, 0.75]) -> float:
        """策略7: 分位数平均"""
        if not predictions:
            return None
        
        quantile_values = [np.quantile(predictions, q) for q in quantiles]
        return np.mean(quantile_values)
    
    def strategy_ensemble_voting(self, predictions: List[float], bins: int = 10) -> float:
        """策略8: 集成投票（将预测值分桶后投票）"""
        if not predictions:
            return None
        
        # 将预测值分桶
        min_val, max_val = min(predictions), max(predictions)
        if min_val == max_val:
            return min_val
        
        bin_width = (max_val - min_val) / bins
        bin_counts = Counter()
        
        for pred in predictions:
            bin_idx = min(int((pred - min_val) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1
        
        # 找到票数最多的桶
        most_common_bin = bin_counts.most_common(1)[0][0]
        bin_start = min_val + most_common_bin * bin_width
        bin_end = min_val + (most_common_bin + 1) * bin_width
        
        # 返回该桶内预测值的平均
        bin_predictions = [p for p in predictions if bin_start <= p < bin_end]
        return np.mean(bin_predictions) if bin_predictions else (bin_start + bin_end) / 2
    
    def strategy_adaptive_trimming(self, predictions: List[float]) -> float:
        """策略9: 自适应截尾（基于变异系数）"""
        if len(predictions) < 3:
            return np.mean(predictions) if predictions else None
        
        predictions = np.array(predictions)
        cv = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0
        
        # 根据变异系数调整截尾比例
        if cv < 0.1:  # 低变异，不截尾
            trim_ratio = 0.0
        elif cv < 0.2:  # 中等变异，轻微截尾
            trim_ratio = 0.1
        else:  # 高变异，大幅截尾
            trim_ratio = 0.3
        
        return self.strategy_trimmed_mean(predictions.tolist(), trim_ratio)

    # For our dataset
    # def strategy_bayesian_average(self, predictions: List[float], prior_mean: float = 108.0, prior_weight: float = 0.9) -> float:
    #     """策略10: 贝叶斯平均（结合先验信息）"""
    #     if not predictions:
    #         return prior_mean
        
    #     sample_mean = np.mean(predictions)
    #     sample_size = len(predictions)
        
    #     # 贝叶斯更新
    #     posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
    #     return posterior_mean
    
    # def strategy_bayesian_average_withRAG(self, predictions: List[float], result: Dict, prior_weight: float = 0.9, default_prior_mean: float = 108.0) -> float:
    #     """策略10-RAG: 贝叶斯平均（使用RAG检索的reference_duration_avg作为先验信息，适用于our dataset参数设置）"""
    #     if not predictions:
    #         # 如果没有预测值，使用reference_duration_avg或默认值
    #         reference_duration_avg = result.get('reference_duration_avg')
    #         if reference_duration_avg is not None and reference_duration_avg > 0:
    #             return reference_duration_avg
    #         return default_prior_mean
    #     
    #     # 从RAG结果中获取reference_duration_avg作为先验均值
    #     reference_duration_avg = result.get('reference_duration_avg')
    #     if reference_duration_avg is None or reference_duration_avg <= 0:
    #         # 如果reference_duration_avg无效，使用默认值
    #         prior_mean = default_prior_mean
    #     else:
    #         prior_mean = reference_duration_avg
    #     
    #     sample_mean = np.mean(predictions)
    #     sample_size = len(predictions)
    #     
    #     # 贝叶斯更新
    #     posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
    #     return posterior_mean

    # For MMSDP dataset
    def strategy_bayesian_average(self, predictions: List[float], prior_mean: float = 140.0, prior_weight: float = 0.5) -> float:
        """策略10: 贝叶斯平均（结合先验信息）"""
        if not predictions:
            return prior_mean
        
        sample_mean = np.mean(predictions)
        sample_size = len(predictions)
        
        # 贝叶斯更新
        posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
        return posterior_mean
    
    def strategy_bayesian_average_withRAG(self, predictions: List[float], result: Dict, prior_weight: float = 0.5, default_prior_mean: float = 140.0) -> float:
        """策略10-RAG: 贝叶斯平均（使用RAG检索的reference_duration_avg作为先验信息）"""
        if not predictions:
            # 如果没有预测值，使用reference_duration_avg或默认值
            reference_duration_avg = result.get('reference_duration_avg')
            if reference_duration_avg is not None and reference_duration_avg > 0:
                return reference_duration_avg
            return default_prior_mean
        
        # 从RAG结果中获取reference_duration_avg作为先验均值
        reference_duration_avg = result.get('reference_duration_avg')
        if reference_duration_avg is None or reference_duration_avg <= 0:
            # 如果reference_duration_avg无效，使用默认值
            prior_mean = default_prior_mean
        else:
            prior_mean = reference_duration_avg
        
        sample_mean = np.mean(predictions)
        sample_size = len(predictions)
        
        # 贝叶斯更新
        posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
        return posterior_mean
    
    def strategy_dynamic_bayesian_average(self, predictions: List[float], result: Dict, base_prior_weight: float = 0.9) -> float:
        """策略11: 动态贝叶斯平均（基于训练集统计的先验信息）"""
        if not predictions:
            return None
        
        # 获取当前测试样本的科室和手术等级信息
        surgery_id = result.get('surgery_id', '')
        if not surgery_id:
            # 如果没有surgery_id，回退到简单平均
            return np.mean(predictions)
        
        # 从训练集统计中获取对应的先验信息（分层匹配）
        prior_info = self._get_prior_info_from_training_stats(surgery_id)
        if prior_info is None:
            # 使用固定先验值
            prior_mean = 108.0
            prior_weight = base_prior_weight
        else:
            prior_mean = prior_info['median']
            # 根据匹配样本数量动态调整先验权重
            match_count = prior_info['count']
            if match_count >= 200:
                base_weight = 2.5  # 高置信度
            elif match_count >= 100:
                base_weight = 2.0  # 中等置信度
            elif match_count >= 50:
                base_weight = 1.5  # 低置信度
            else:
                base_weight = 0.9  # 很低置信度
            
            # 一致性检查：如果先验与当前预测差异过大，降低权重
            current_mean = np.mean(predictions)
            relative_diff = abs(prior_mean - current_mean) / current_mean if current_mean > 0 else 0
            
            if relative_diff > 0.5:  # 差异超过50%
                consistency_factor = 0.8
            elif relative_diff > 0.3:  # 差异超过30%
                consistency_factor = 0.9
            else:
                consistency_factor = 1.0  # 保持原权重
            
            prior_weight = base_weight * consistency_factor
        
        sample_mean = np.mean(predictions)
        sample_size = len(predictions)
        
        # 贝叶斯更新
        posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
        return posterior_mean
    
    def strategy_current_implementation(self, predictions: List[float]) -> float:
        """策略12: 当前实现策略（去掉最高最低后取平均）"""
        if not predictions:
            return None
        
        if len(predictions) >= 3:
            # 如果有3个或以上成功预测，去掉最高和最低值后取平均
            sorted_predictions = sorted(predictions)
            trimmed_predictions = sorted_predictions[1:-1]  # 去掉最高和最低
            return np.mean(trimmed_predictions)
        else:
            # 如果只有1-2个成功预测，直接取平均
            return np.mean(predictions)

    def strategy_weighted_bayesian_average(self, predictions: List[float], result: Dict) -> float:
        """策略15: 加权贝叶斯平均（基于预测质量调整权重）"""
        if not predictions:
            return None
        
        # 计算预测的稳定性
        cv = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0
        
        # 根据稳定性调整先验权重
        if cv < 0.1:
            prior_weight = 0.8  
        elif cv > 0.3: 
            prior_weight = 1.1
        else:
            prior_weight = 0.9

        sample_mean = np.mean(predictions)
        if sample_mean > 300:
            prior_mean = 115
        elif sample_mean >= 120 and sample_mean <= 240:
            prior_mean = 110
        else:
            prior_mean = 108.0

        sample_size = len(predictions)
        
        return (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
 
    def apply_strategy(self, strategy_name: str, result: Dict) -> float:
        """应用指定策略计算最终预测值"""
        predictions = self._get_valid_predictions(result)
        if not predictions:
            return None
        
        # 获取温度信息（如果可用）
        temperatures = result.get('all_temperature', [])
        
        if strategy_name == 'simple_average':
            return self.strategy_simple_average(predictions)
        elif strategy_name == 'trimmed_mean':
            return self.strategy_trimmed_mean(predictions)
        elif strategy_name == 'median':
            return self.strategy_median(predictions)
        # elif strategy_name == 'weighted_average':
        #     return self.strategy_weighted_average(predictions)
        elif strategy_name == 'robust_mean':
            return self.strategy_robust_mean(predictions)
        elif strategy_name == 'confidence_weighted':
            return self.strategy_confidence_weighted(predictions, temperatures)
        elif strategy_name == 'quantile_average':
            return self.strategy_quantile_average(predictions)
        elif strategy_name == 'ensemble_voting':
            return self.strategy_ensemble_voting(predictions)
        elif strategy_name == 'adaptive_trimming':
            return self.strategy_adaptive_trimming(predictions)
        elif strategy_name == 'bayesian_average':
            return self.strategy_bayesian_average(predictions)
        elif strategy_name == 'bayesian_average_withRAG':
            return self.strategy_bayesian_average_withRAG(predictions, result)
        # elif strategy_name == 'dy_bayesian_average':
        #     return self.strategy_dynamic_bayesian_average(predictions, result)
        elif strategy_name == 'current_method*':
            return self.strategy_current_implementation(predictions)
        # elif strategy_name == 'weighted_bayesian_average':
        #     return self.strategy_weighted_bayesian_average(predictions, result)
        else:
            raise ValueError(f"未知策略: {strategy_name}")
    
    def evaluate_strategy(self, strategy_name: str) -> Dict:
        """评估单个策略的性能"""
        print(f"\n评估策略: {strategy_name}")
        
        # 应用策略重新计算预测结果
        new_results = []
        for result in self.results: # result是每个患者的多次预测结果
            new_prediction = self.apply_strategy(strategy_name, result)
            
            # 创建新的结果记录
            new_result = result.copy()
            new_result['predicted_duration'] = new_prediction
            new_result['success'] = new_prediction is not None
            new_results.append(new_result)
        
        # 计算性能指标
        metrics = calculate_metrics(new_results)
        return metrics
    
    def evaluate_all_strategies(self) -> Dict[str, Dict]:
        """评估所有策略的性能"""
        strategy_names = [
            'current_method*',  # 当前实现策略（作为基准）
            'simple_average',
            'trimmed_mean', 
            'median',
            'weighted_average',
            'robust_mean',
            'confidence_weighted',
            'quantile_average',
            'ensemble_voting',
            'adaptive_trimming',
            'bayesian_average',
            'bayesian_average_withRAG',  # 使用RAG检索结果的贝叶斯平均
            'dy_bayesian_average',  # 动态贝叶斯平均
            'weighted_bayesian_average',  # 加权贝叶斯平均
        ]
        
        results = {}
        for strategy_name in strategy_names:
            try:
                metrics = self.evaluate_strategy(strategy_name)
                if metrics:
                    results[strategy_name] = metrics
                    print(f"策略 {strategy_name} 评估完成")
                else:
                    print(f"策略 {strategy_name} 评估失败")
            except Exception as e:
                print(f"策略 {strategy_name} 评估出错: {e}")
        
        return results
    
    def compare_strategies(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """比较所有策略的性能"""
        comparison_data = []
        
        for strategy_name, metrics in results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE'],
                'Success_Rate': metrics['success_rate']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 按MAE排序（越小越好）
        df = df.sort_values('MAE')
        
        return df
    
    def print_comparison(self, df: pd.DataFrame):
        """打印策略比较结果"""
        print("\n" + "="*80)
        print("多次预测聚合策略性能比较")
        print("="*80)
        print(f"{'策略名称':<30} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<8} {'成功率':<8}")
        print("-"*80)
        
        for _, row in df.iterrows():
            print(f"{row['Strategy']:<30} {row['MAE']:<8.2f} {row['RMSE']:<8.2f} {row['R2']:<8.4f} {row['MAPE']:<8.2f} {row['Success_Rate']:<8.2f}")
        
        print("\n最佳策略 (按MAE排序):")
        best_strategy = df.iloc[0]
        print(f"  策略: {best_strategy['Strategy']}")
        print(f"  MAE: {best_strategy['MAE']:.2f}")
        print(f"  RMSE: {best_strategy['RMSE']:.2f}")
        print(f"  R²: {best_strategy['R2']:.4f}")
        print(f"  MAPE: {best_strategy['MAPE']:.2f}%")
        print(f"  成功率: {best_strategy['Success_Rate']:.2f}%")
    
    def save_results(self, results: Dict[str, Dict], output_file: str = 'ensemble_strategies_results.json'):
        """保存评估结果"""
        # 转换numpy类型为Python原生类型
        serializable_results = {}
        for strategy_name, metrics in results.items():
            serializable_results[strategy_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[strategy_name][key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    serializable_results[strategy_name][key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    serializable_results[strategy_name][key] = float(value)
                else:
                    serializable_results[strategy_name][key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存到: {output_file}")

def main():
    """主函数"""
    # 设置结果文件路径
    results_file = './predict_results/rag_8shot_Flat_5times_priority_nodoctor_huatuogpt_prediction_results.json'
    # results_file = './predict_results/rag_8shot_Flat_5times_priority_nodoctor_deepseek-r1_prediction_results.json'
    # results_file = './predict_results/rag_8shot_Flat_5times_priority_nodoctor_qwen3_prediction_results.json'
    # results_file = "./predict_results/rag_8shot_Flat_5times_qwen3_4b_prediction_results.json"
    # results_file = "./predict_results/rag_8shot_Flat_5times_qwen3_14b_prediction_results.json"
    # results_file = "./predict_results/rag_8shot_Flat_5times_qwen3_32b_prediction_results copy.json"
    # results_file = "./predict_results/ablation_rag_8shot_Flat_10times_qwen3_prediction_results.json"
    results_file = "/home/likx/time_series_forecasting/surgical_duration_prediction/Multimodal_SDP_rag_8shot_Flat_5times_qwen3_prediction_results.json"

    # results_file = '/home/likx/time_series_forecasting/surgical_duration_prediction/rag_8shot_Flat_10times_with_priority_nodoctor_huatuogpt_prediction_results.json'

    # 检查文件是否存在
    if not os.path.exists(results_file):
        print(f"错误: 结果文件不存在: {results_file}")
        print("请确保文件路径正确")
        return
    
    # 加载训练集和测试集数据（用于动态贝叶斯平均）
    try:
        from dataset_provider.dataset4LLM_utils import read_dataset
        
        # 加载数据集
        root_path = "/home/likx/time_series_forecasting/SDP_DATASET"
        train_df, val_df, test_df = read_dataset(os.path.join(root_path, "train.pkl")), \
                                   read_dataset(os.path.join(root_path, "val.pkl")), \
                                   read_dataset(os.path.join(root_path, "test.pkl"))
        
        print(f"成功加载数据集: 训练集{len(train_df)}条, 测试集{len(test_df)}条")
        
        # 创建评估器
        evaluator = EnsembleStrategies(results_file, train_df, test_df)
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("将使用不包含动态贝叶斯平均的评估器")
        
        # 创建评估器（不包含训练集信息）
        evaluator = EnsembleStrategies(results_file)
    
    if not evaluator.results:
        print("没有可用的预测结果")
        return
    
    print(f"开始评估 {len(evaluator.results)} 个样本的多次预测聚合策略")
    
    # 评估所有策略
    results = evaluator.evaluate_all_strategies()
    
    if not results:
        print("没有成功的策略评估")
        return
    
    # 比较策略性能
    comparison_df = evaluator.compare_strategies(results)
    
    # 打印比较结果
    evaluator.print_comparison(comparison_df)
    
    # 保存结果
    # evaluator.save_results(results)
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()
