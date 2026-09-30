"""
计算LLaMA-Factory评估结果中各模型的性能指标

读取各模型目录下的 generated_predictions.jsonl，计算 MAE、RMSE、R²、MAPE、MedAE、PAE。
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 与 compute_metrics / adjust_prediction_mae 默认一致
_DEFAULT_EPS = 1e-6
_DEFAULT_PAE_PCT = 20.0


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot（与 sklearn r2_score 一致）。"""
    n = len(y_true)
    if n == 0:
        return float("nan")
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    return 1.0 - ss_res / ss_tot


def _mape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    """MAPE（%）：mean(|pred-y| / max(|y|, eps)) * 100。"""
    den = np.maximum(np.abs(y_true), eps)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / den))


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true)))


def _pae_pct(
    y_true: np.ndarray, y_pred: np.ndarray, rel_threshold: float, eps: float
) -> float:
    """PAE（%）：相对真实值误差 ≤ rel_threshold 的样本占比 * 100。"""
    den = np.maximum(np.abs(y_true), eps)
    re = np.abs(y_pred - y_true) / den
    return float(100.0 * np.mean(re <= rel_threshold))


def extract_number(value):
    """
    从字符串中提取数字值
    
    Args:
        value: 可能是字符串、数字或包含换行符的字符串
    
    Returns:
        float: 提取的数值，如果无法提取则返回None
    """
    if value is None:
        return None
    
    # 如果是数字类型，直接返回
    if isinstance(value, (int, float)):
        return float(value)
    
    # 如果是字符串，去除空格和换行符，然后尝试提取数字
    if isinstance(value, str):
        value = value.strip().replace('\n', '').replace('\r', '')
        try:
            # 尝试直接转换
            return float(value)
        except (ValueError, TypeError):
            # 如果失败，尝试提取数字（处理类似 "100.0" 或 " 100.0" 的情况）
            import re
            numbers = re.findall(r'-?\d+\.?\d*', value)
            if numbers:
                try:
                    return float(numbers[0])
                except (ValueError, TypeError):
                    return None
            return None
    
    return None


def load_predictions_from_jsonl(jsonl_path):
    """
    从jsonl文件中加载预测结果
    
    Args:
        jsonl_path: jsonl文件路径
    
    Returns:
        tuple: (predictions, labels, valid_count, total_count)
            predictions: 预测值列表
            labels: 真实值列表
            valid_count: 有效样本数
            total_count: 总样本数
    """
    predictions = []
    labels = []
    total_count = 0
    valid_count = 0
    
    if not os.path.exists(jsonl_path):
        print(f"警告: 文件不存在 {jsonl_path}")
        return predictions, labels, valid_count, total_count
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # 提取predict和label
                pred_value = extract_number(data.get('predict'))
                label_value = extract_number(data.get('label'))
                
                # 只保留有效的预测结果（两者都不为None且大于0）
                if pred_value is not None and label_value is not None and pred_value > 0 and label_value > 0:
                    predictions.append(pred_value)
                    labels.append(label_value)
                    valid_count += 1
                else:
                    # 记录无效样本
                    if pred_value is None or pred_value <= 0:
                        pass  # 预测值无效
                    if label_value is None or label_value <= 0:
                        pass  # 标签值无效
                        
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    return predictions, labels, valid_count, total_count


def calculate_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    eps: float = _DEFAULT_EPS,
    pae_pct: float = _DEFAULT_PAE_PCT,
):
    """
    计算性能指标（MAE / RMSE / R2 / MAPE / MedAE / PAE，公式与 compute_metrics 一致）。
    
    Args:
        y_true: 真实值列表或数组
        y_pred: 预测值列表或数组
        eps: 相对误差分母下界
        pae_pct: PAE 阈值（百分比），如 20 表示 20%
    
    Returns:
        dict: 包含各种性能指标的字典
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "valid_count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "MedAE": np.nan,
            "PAE": np.nan,
        }

    rel_threshold = pae_pct / 100.0
    return {
        "valid_count": int(len(y_true)),
        "MAE": _mae(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "R2": _r2(y_true, y_pred),
        "MAPE": _mape_pct(y_true, y_pred, eps),
        "MedAE": _medae(y_true, y_pred),
        "PAE": _pae_pct(y_true, y_pred, rel_threshold, eps),
    }


def analyze_all_models(base_dir):
    """
    分析base_dir目录下所有模型的性能
    
    Args:
        base_dir: 基础目录路径（包含各模型子目录）
    
    Returns:
        pd.DataFrame: 包含各模型性能指标的DataFrame
    """
    results = []
    
    # 查找所有包含generated_predictions.jsonl的目录
    model_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            jsonl_path = os.path.join(item_path, 'generated_predictions.jsonl')
            if os.path.exists(jsonl_path):
                model_dirs.append((item, jsonl_path))
    
    if not model_dirs:
        print(f"警告: 在 {base_dir} 下没有找到包含generated_predictions.jsonl的目录")
        return pd.DataFrame()
    
    print(f"找到 {len(model_dirs)} 个模型目录，开始分析...\n")
    
    # 分析每个模型
    for model_name, jsonl_path in sorted(model_dirs):
        print(f"正在分析模型: {model_name}")
        print(f"  文件路径: {jsonl_path}")
        
        # 加载预测结果
        predictions, labels, valid_count, total_count = load_predictions_from_jsonl(jsonl_path)
        
        print(f"  总样本数: {total_count}")
        print(f"  有效样本数: {valid_count}")
        
        if valid_count == 0:
            print(f"  警告: 没有有效样本，跳过该模型\n")
            results.append({
                'model_name': model_name,
                'total_count': total_count,
                'valid_count': 0,
                'success_rate': 0.0,
                'MAE': np.nan,
                'RMSE': np.nan,
                'MAPE': np.nan,
                'R2': np.nan,
                'MedAE': np.nan,
                'PAE': np.nan,
            })
            continue
        
        # 计算性能指标
        metrics = calculate_metrics(labels, predictions)
        
        success_rate = (valid_count / total_count * 100) if total_count > 0 else 0.0
        
        result = {
            'model_name': model_name,
            'total_count': total_count,
            'valid_count': valid_count,
            'success_rate': success_rate,
            **metrics
        }
        results.append(result)
        
        # 打印结果
        print(f"  性能指标:")
        print(f"    MAE: {metrics['MAE']:.2f} 分钟")
        print(f"    RMSE: {metrics['RMSE']:.2f} 分钟")
        print(f"    R²: {metrics['R2']:.4f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")
        print(f"    MedAE: {metrics['MedAE']:.2f} 分钟")
        print(f"    PAE: {metrics['PAE']:.2f}% (≤{_DEFAULT_PAE_PCT:g}% vs |y|)")
        print()
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='计算LLaMA-Factory模型性能指标')
    parser.add_argument('--base_dir', type=str, # DeepSeek-R1-8B-Distill, HuatuoGPT-o1-7B, Qwen3-8B-Thinking, Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410, BLOOM-7B1,Gemma-7B-Instruct
                       default='/home/likx/time_series_forecasting/LLaMA-Factory/saves/BLOOM-7B1/lora', 
                       help='模型目录基础路径')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出CSV文件路径（可选）')
    args = parser.parse_args()
    
    # 分析所有模型
    results_df = analyze_all_models(args.base_dir)
    
    if results_df.empty:
        print("没有找到任何有效的模型结果")
        return
    
    # 打印汇总结果
    print("=" * 80)
    print("模型性能指标汇总")
    print("=" * 80)
    print()
    
    # 格式化打印表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(results_df.to_string(index=False))
    print()
    
    # 保存到CSV文件
    if args.output_file:
        results_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {args.output_file}")
    else:
        # 默认保存到当前目录
        default_output = os.path.join(args.base_dir, 'model_performance_metrics.csv')
        results_df.to_csv(default_output, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {default_output}")
    
    print()
    print("=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

