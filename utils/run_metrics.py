import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_EPS = 1e-6
_PAE_PCT = 20.0


def _mape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    den = np.maximum(np.abs(y_true), eps)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / den))


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true)))


def _pae_pct(
    y_true: np.ndarray, y_pred: np.ndarray, rel_threshold: float, eps: float
) -> float:
    den = np.maximum(np.abs(y_true), eps)
    re = np.abs(y_pred - y_true) / den
    return float(100.0 * np.mean(re <= rel_threshold))

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_model(y_true, y_pred):
    """
    评估模型性能（MAE / MSE / RMSE / R2 / MAPE / MedAE / PAE）。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "PAE": np.nan,
            "MAPE": np.nan,
            "MedAE": np.nan,
        }
    mse = float(mean_squared_error(y_true, y_pred))
    rel_th = _PAE_PCT / 100.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": _mape_pct(y_true, y_pred, _EPS),
        "MedAE": _medae(y_true, y_pred),
        "PAE": _pae_pct(y_true, y_pred, rel_th, _EPS),
    }

def load_prediction_results(json_file_path):
    """
    加载预测结果JSON文件
    
    Args:
        json_file_path (str): JSON文件路径
    
    Returns:
        list: 预测结果列表
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # print(f"成功加载预测结果: {len(results)} 个样本")
        return results
    except FileNotFoundError:
        print(f"错误: 未找到文件 {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"错误: 加载文件时出错 - {e}")
        return None

def calculate_metrics(results):
    """
    计算模型性能指标
    
    Args:
        results (list): 预测结果列表
    
    Returns:
        dict: 性能指标字典
    """
    if not results:
        print("错误: 没有可用的预测结果")
        return None
    
    # 分离成功和失败的预测
    successful_predictions = [r for r in results if r['success'] and r['predicted_duration'] is not None]
    failed_predictions = [r for r in results if not r['success'] or r['predicted_duration'] is None]
    
    # print(f"成功预测: {len(successful_predictions)} 个")
    # print(f"失败预测: {len(failed_predictions)} 个")
    # print(f"成功率: {len(successful_predictions)/len(results)*100:.2f}%")
    
    if not successful_predictions:
        print("错误: 没有成功的预测结果")
        return None
    
    # 提取真实值和预测值
    y_true = np.array([r['ground_true'] for r in successful_predictions])
    y_pred = np.array([r['predicted_duration'] for r in successful_predictions])
    
    # 计算性能指标
    metrics = evaluate_model(y_true, y_pred)
    
    # 添加额外统计信息
    metrics['total_samples'] = len(results)
    metrics['successful_samples'] = len(successful_predictions)
    metrics['failed_samples'] = len(failed_predictions)
    metrics['success_rate'] = len(successful_predictions) / len(results) * 100
    
    # 计算温度统计（如果有的话）
    if successful_predictions and 'final_temperature' in successful_predictions[0]:
        temperatures = [r['final_temperature'] for r in successful_predictions]
        metrics['avg_temperature'] = np.mean(temperatures)
        metrics['temperature_std'] = np.std(temperatures)
    else:
        metrics['avg_temperature'] = 0.0
        metrics['temperature_std'] = 0.0
    
    return metrics

def analyze_failed_predictions(results):
    """
    分析失败的预测
    
    Args:
        results (list): 预测结果列表
    
    Returns:
        dict: 失败分析结果
    """
    failed_predictions = [r for r in results if not r['success'] or r['predicted_duration'] is None]
    
    if not failed_predictions:
        return {"message": "没有失败的预测"}
    
    # 分析失败原因
    failure_analysis = {
        'total_failures': len(failed_predictions),
        'failure_rate': len(failed_predictions) / len(results) * 100,
        'retry_distribution': {},
        'temperature_distribution': {},
        'sample_outputs': []
    }
    
    # 统计温度分布（如果有的话）
    if failed_predictions and 'final_temperature' in failed_predictions[0]:
        temperatures = [r['final_temperature'] for r in failed_predictions]
        for temp in set(temperatures):
            failure_analysis['temperature_distribution'][f'temp_{temp}'] = temperatures.count(temp)
    
    # 保存一些失败的模型输出示例
    failure_analysis['sample_outputs'] = [
        {
            'surgery_id': r['surgery_id'],
            'model_output': r['model_output'][:200] + '...' if len(r['model_output']) > 200 else r['model_output']
        }
        for r in failed_predictions[:5]  # 只保存前5个示例
    ]
    
    return failure_analysis

def save_metrics_report(metrics, failure_analysis, output_file='llm_metrics_report.json'):
    """
    保存性能指标报告
    
    Args:
        metrics (dict): 性能指标
        failure_analysis (dict): 失败分析
        output_file (str): 输出文件名
    """
    report = {
        'performance_metrics': metrics,
        'failure_analysis': failure_analysis,
        'summary': {
            'model_name': 'LLM4SDP',
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': metrics['total_samples'],
            'success_rate': f"{metrics['success_rate']:.2f}%",
            'best_metric': 'MAPE' if metrics['MAPE'] < 50 else 'MAE',
            'best_value': metrics['MAPE'] if metrics['MAPE'] < 50 else metrics['MAE']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"性能指标报告已保存到: {output_file}")

def print_metrics_summary(metrics):
    """
    打印性能指标摘要
    
    Args:
        metrics (dict): 性能指标字典
    """
    print(f"  总样本数: {metrics['total_samples']}, 成功预测: {metrics['successful_samples']}")
    print(f"  MAE (平均绝对误差): {metrics['MAE']:.2f}")
    print(f"  RMSE (均方根误差): {metrics['RMSE']:.2f}")
    print(f"  R² (决定系数): {metrics['R2']:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {metrics['MAPE']:.2f}%")
    print(f"  MedAE (绝对误差中位数): {metrics['MedAE']:.2f}")
    print(f"  PAE (≤{_PAE_PCT:g}% vs |y|): {metrics['PAE']:.2f}%")
    
    # print(f"\n重试统计:")
    # print(f"  平均重试次数: {metrics['avg_retries']:.2f}")
    # print(f"  最大重试次数: {metrics['max_retries']}")
    # print(f"  最小重试次数: {metrics['min_retries']}")
    
    # print(f"\n温度统计:")
    # print(f"  平均温度: {metrics['avg_temperature']:.3f}")
    # print(f"  温度标准差: {metrics['temperature_std']:.3f}")

def main():
    """
    主函数：计算LLM4SDP模型性能指标
    """
    # 设置文件路径
    # json_file = 'prediction_results.json'
    # json_file = 'fixed_5shot_1tests_prediction_results.json'
    json_file = 'fixed_5shot_10tests_prediction_results.json'
    
    
    # 加载预测结果
    results = load_prediction_results(json_file)
    if results is None:
        return
    
    # 计算性能指标
    metrics = calculate_metrics(results)
    if metrics is None:
        return
    
    # 分析失败预测
    # print("\n分析失败预测...")
    # failure_analysis = analyze_failed_predictions(results)
    
    # 打印结果摘要
    print_metrics_summary(metrics)
    
    # # 打印失败分析
    # if failure_analysis.get('total_failures', 0) > 0:
    #     print(f"\n失败预测分析:")
    #     print(f"  失败数量: {failure_analysis['total_failures']}")
    #     print(f"  失败率: {failure_analysis['failure_rate']:.2f}%")
    #     print(f"  重试分布: {failure_analysis['retry_distribution']}")
    #     print(f"  温度分布: {failure_analysis['temperature_distribution']}")
    # else:
    #     print(f"\n所有预测都成功！")
    
    # 保存详细报告
    # save_metrics_report(metrics, failure_analysis)
    
    # print(f"\n评估完成！")

if __name__ == "__main__":
    main()
