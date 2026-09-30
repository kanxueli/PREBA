import json
import os
import re
import sys
import time
import random
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from transformers.utils.versions import require_version
from dataset_provider.dataset4LLM_utils import read_dataset
from utils.prompt_constructor import (
    System_prompt,
    System_prompt_MMSDP,
    System_prompt_INSPIRE,
    System_prompt_MOVER_EPIC,
    System_prompt_MOVER_SIS,
    fixed_reference_instance_construction,
    rag_reference_instance_construction,
    to_be_predicted_instance_construction,
    full_prompt_construction,
    safe_get_value,
)
from utils.prior_hint_generator import PriorHintGenerator
from utils.run_metrics import calculate_metrics, print_metrics_summary
from utils.rag_database import FAISSVectorDatabase
from utils.rag_retriever import RAGRetriever, RAGDataMapper
from utils.rag_weights import WEIGHT_SCHEMES, set_weight_scheme, faiss_weight_suffix
from utils.rag_post_processor import RAGPostProcessor
from utils.ensemble_strategies import EnsembleStrategies

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def get_case_info(row, dataset_name=''):
    """
    根据数据集类型获取案例的关键信息
    
    Args:
        row: 数据行（pandas Series）
        dataset_name: 数据集名称
    
    Returns:
        dict: 包含案例关键信息的字典
    """
    if dataset_name == 'Multimodal_SDP':
        # MMSDP数据集使用英文字段名
        return {
            'department': safe_get_value(row, 'department', '未知科室'),
            'surgery': safe_get_value(row, 'procedures', '未知手术'),
            'duration': safe_get_value(row, 'duration', '未知时长'),
            'diagnosis': safe_get_value(row, 'diagnoses', '无'),
            'level': safe_get_value(row, 'level', '目前无评级'),
            'asa': safe_get_value(row, 'assessment_asa', '目前无评级'),
            'instrument': None,  # MMSDP没有此字段
            'nyha': None  # MMSDP没有此字段
        }
    if dataset_name == 'INSPIRE':
        return {
            'department': safe_get_value(row, 'department.1', '未知科室'),
            'surgery': safe_get_value(row, 'ICD', '未知手术'),
            'duration': safe_get_value(row, 'duration', '未知时长'),
            'diagnosis': safe_get_value(row, 'diagnoses_text', '无'),
            'level': None,
            'asa': safe_get_value(row, 'assessment_asa', '目前无评级'),
            'instrument': None,
            'nyha': None
        }
    if dataset_name == 'MOVER_EPIC':
        return {
            'department': None,
            'surgery': safe_get_value(row, 'procedure_name', '未知手术'),
            'duration': safe_get_value(row, 'duration', '未知时长'),
            'diagnosis': safe_get_value(row, 'diagnosis_names', '无'),
            'level': None,
            'asa': safe_get_value(row, 'assessment_asa', '目前无评级'),
            'instrument': None,
            'nyha': None
        }
    if dataset_name == 'MOVER_SIS':
        return {
            'department': None,
            'surgery': safe_get_value(row, 'procedure_name', '未知手术'),
            'duration': safe_get_value(row, 'duration', '未知时长'),
            'diagnosis': safe_get_value(row, 'preop_medications', '无'),
            'level': None,
            'asa': None,
            'instrument': None,
            'nyha': None
        }
    else:
        # 原始数据集使用中文字段名
        return {
            'department': safe_get_value(row, '患者住院科室', '未知科室'),
            'surgery': safe_get_value(row, '拟行手术名称', '未知手术'),
            'duration': safe_get_value(row, '手术间时长', '未知时长'),
            'diagnosis': safe_get_value(row, '主要诊断', '无'),
            'level': safe_get_value(row, '手术等级', '目前无评级'),
            'asa': safe_get_value(row, '总体评估时的ASA分级', '目前无评级'),
            'instrument': safe_get_value(row, '手术器械清点单手术名称', '无'),
            'nyha': safe_get_value(row, '麻醉计划风险评估时患者心功能分级(New York)', '目前无评级')
        }

def format_case_info(info, dataset_name='', include_instrument=True, include_nyha=True):
    """
    格式化案例信息为字符串
    
    Args:
        info: 案例信息字典（来自get_case_info）
        dataset_name: 数据集名称
        include_instrument: 是否包含器械信息
        include_nyha: 是否包含心功能分级信息
    
    Returns:
        str: 格式化后的字符串
    """
    parts = [
        f"科室={info['department']}",
        f"时长={info['duration']}min",
        f"主要诊断={info['diagnosis']}",
        f"拟行手术名称={info['surgery']}",
    ]
    
    if include_instrument and info['instrument'] is not None:
        parts.append(f"手术器械清点单手术名称={info['instrument']}")
    
    parts.extend([
        f"手术等级={info['level']}",
        f"ASA分级={info['asa']}"
    ])
    
    if include_nyha and info['nyha'] is not None:
        parts.append(f"患者心功能分级={info['nyha']}")
    
    return ", ".join(parts)

def predict_single_sample(args_tuple):
    """
    预测单个样本的函数，用于并发处理
    
    Args:
        args_tuple: (index, row, client, model_id, model_type, reference_instance, n_predictions, max_retries, save_prompt_example, use_rag, rag_retriever, rag_mapper, rag_k, rag_post_processor, prior_hint_generator, dataset_name)
    
    Returns:
        dict: 预测结果记录
    """
    index, row, client, model_id, model_type, reference_instance, n_predictions, max_retries, save_prompt_example, use_rag, rag_retriever, rag_mapper, rag_k, rag_post_processor, prior_hint_generator, dataset_name = args_tuple
    
    # 构建待预测实例
    question, ground_true = to_be_predicted_instance_construction(row, dataset_name=dataset_name)
    if ground_true <= 0:
        return None

    # 进行RAG检索相关实例
    if use_rag and rag_retriever is not None and rag_mapper is not None:
        # 打印当前样本信息
        query_info = get_case_info(row, dataset_name)
        query_str = format_case_info(query_info, dataset_name, include_instrument=True, include_nyha=True)
        print(f"查询样本 {index}: {query_str}")
        
        # 使用RAG检索相关案例 - 检索10倍数量用于后处理筛选
        try:
            # # 检索 n_predictions * rag_k * 10 个案例，用于后处理筛选
            # target_case_count = min(n_predictions * rag_k, 2 * rag_k)
            # target_case_count = min(n_predictions * rag_k, int(1.5 * rag_k+0.5))
            # total_retrieval_count = min(target_case_count * 30, 150)
            target_case_count = rag_k * 2
            total_retrieval_count = target_case_count
            retrieval_results = rag_retriever.retrieve_similar_cases(row, k=total_retrieval_count)
            if retrieval_results:
                # 将检索结果映射回原始数据格式（已按相似度排序）
                retrieved_cases = rag_mapper.map_to_original_data(retrieval_results)
                
                # 提取相似度分数
                similarity_scores = [result['similarity'] for result in retrieval_results]

                # # 循环打印检索到的案例基本信息和相似度分数
                print(f"样本 {index} 检索到 {len(retrieved_cases)} 个案例:")
                for i in range(min(len(retrieved_cases), 10)):  # 最多打印前10个
                    case = retrieved_cases.iloc[i]
                    case_info = get_case_info(case, dataset_name)
                    case_str = format_case_info(case_info, dataset_name, include_instrument=True, include_nyha=False)
                    print(f"  案例 {i+1}: 相似度={similarity_scores[i]:.4f}, {case_str}")
                
                # 使用后处理器筛选高质量案例
                if rag_post_processor is not None:
                    filtered_cases, filtered_scores = rag_post_processor.process_retrieved_cases(
                        row, retrieved_cases, similarity_scores, target_case_count
                    )
                    
                    if len(filtered_cases) > 0:
                        all_retrieved_cases = filtered_cases
                        all_similarity_scores = filtered_scores
                        print(f"样本 {index} 后处理筛选完成，最终获得 {len(filtered_cases)} 个高质量案例")
                    else:
                        # 如果后处理筛选结果为空，使用原始检索结果的前target_case_count个
                        print(f"样本 {index} 后处理筛选无结果，使用原始检索的前{target_case_count}个案例")
                        all_retrieved_cases = retrieved_cases.head(target_case_count)
                        all_similarity_scores = similarity_scores[:target_case_count]
                else:
                    # 如果没有后处理器，使用原始检索结果的前target_case_count个
                    all_retrieved_cases = retrieved_cases.head(target_case_count)
                    all_similarity_scores = similarity_scores[:target_case_count]
            else:
                # 如果RAG检索失败，使用固定参考实例
                print(f"样本 {index} RAG检索无结果，使用固定参考实例")
                all_retrieved_cases = None
                all_similarity_scores = None
        except Exception as e:
            print(f'样本 {index} RAG检索失败: {e}，使用固定参考实例')
            all_retrieved_cases = None
            all_similarity_scores = None
    else:
        # 使用固定参考实例
        all_retrieved_cases = None
        all_similarity_scores = None

    # 先验提示生成（传入检索案例以添加统计信息）
    prior_hint = prior_hint_generator.generate_prior_hint(row, all_retrieved_cases) if prior_hint_generator else ""
    
    # Multiple predictions averaging    
    all_predictions = []  # Store all successful predictions
    all_temperatures = []  # Store all used temperatures
    all_outputs = []  # Store all model outputs
    
    # 模型预测
    # all_retrieved_cases = all_retrieved_cases.head(2*rag_k)
    # all_similarity_scores = all_similarity_scores[:2*rag_k]
    for pred_idx in range(n_predictions):
        predicted_duration = None
        predicted_text = None
        temperature = 0.0 if pred_idx == 0 else round(random.uniform(0.1, 0.3), 2)
        
        # 为每次预测选择不同的参考实例
        if all_retrieved_cases is not None and len(all_retrieved_cases) >= rag_k:
            # 从检索到的案例中随机选择rag_k个
            if len(all_retrieved_cases) > rag_k:
                if pred_idx == 0:
                    # 第一次预测：按相似度取前 rag_k 个
                    selected_cases = all_retrieved_cases.head(rag_k)
                    selected_scores = all_similarity_scores[:rag_k]
                else:
                    # 随机选择rag_k个案例
                    selected_indices = random.sample(range(len(all_retrieved_cases)), rag_k)
                    selected_cases = all_retrieved_cases.iloc[selected_indices]
                    selected_scores = [all_similarity_scores[i] for i in selected_indices]
            else:
                # 如果检索到的案例不够，使用所有案例
                selected_cases = all_retrieved_cases
                selected_scores = all_similarity_scores
            
            # 构建当前预测的参考实例
            current_reference_instance = rag_reference_instance_construction(selected_cases, selected_scores, dataset_name=dataset_name)
        else:
            # 使用固定参考实例
            current_reference_instance = reference_instance
        
        # 构建当前预测的完整prompt
        sdp_query = full_prompt_construction(current_reference_instance, question, prior_hint, dataset_name=dataset_name)

        # 根据数据集类型选择 System_prompt
        if dataset_name == 'Multimodal_SDP':
            current_system_prompt = System_prompt_MMSDP
        elif dataset_name == 'INSPIRE':
            current_system_prompt = System_prompt_INSPIRE
        elif dataset_name == 'MOVER_EPIC':
            current_system_prompt = System_prompt_MOVER_EPIC
        elif dataset_name == 'MOVER_SIS':
            current_system_prompt = System_prompt_MOVER_SIS
        else:
            current_system_prompt = System_prompt
        
        # Save full prompt example to file
        if save_prompt_example and index % 10 == 0 and index < 50:
            full_prompt_content = f"{current_system_prompt}\n\n\n{sdp_query}"
            with open(f'prompt_logs/example_{index}.txt', 'w', encoding='utf-8') as f:
                f.write(full_prompt_content)

        current_messages = [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": sdp_query}
        ]
        
        # Retry for each prediction
        for retry in range(max_retries):
            try:
                # Set different parameters according to model type
                if model_type == 'qwen3' or model_type == 'qwen3_32b':
                    result = client.chat.completions.create(
                        messages=current_messages, 
                        model=model_id, 
                        temperature=temperature,
                        top_p=0.8,
                        presence_penalty=1.5,
                        stream=False
                    )
                elif model_type == 'huatuogpt':
                    result = client.chat.completions.create(
                        messages=current_messages, 
                        model=model_id, 
                        temperature=temperature,
                        top_p=0.9,
                        presence_penalty=0.2,
                        stream=False
                    )
                else:
                    result = client.chat.completions.create(
                        messages=current_messages, 
                        model=model_id, 
                        temperature=temperature
                    )
                # print("###############################result: ", result)
                predicted_text = result.choices[0].message.content
                predicted_duration = extract_duration_from_text(predicted_text)
                
                if predicted_duration is not None:
                    all_predictions.append(predicted_duration)
                    all_temperatures.append(temperature)
                    all_outputs.append(predicted_text)
                    break
                else:
                    # If parsing fails, try different temperatures
                    temperature = round(random.uniform(0.05, 0.4), 2)
                    
            except Exception as e:
                print(f'Sample {index} Prediction {pred_idx + 1} Retry {retry + 1} Error: {e}')
                temperature = round(random.uniform(0.05, 0.4), 2)
        
        if predicted_duration is None:
            print(f'Sample {index} Prediction {pred_idx + 1} Failed, Model Output: {predicted_text}')
    
    # 首先计算检索案例的均值作为先验均值
    prior_mean = None
    if all_retrieved_cases is not None and len(all_retrieved_cases) > 0:
        # 提取时长字段
        if dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
            duration_field = 'duration'
        else:
            duration_field = '手术间时长'
        
        if duration_field in all_retrieved_cases.columns:
            durations = all_retrieved_cases[duration_field].values
            valid_durations = durations[~np.isnan(durations)]
            valid_durations = valid_durations[valid_durations > 0]
            if len(valid_durations) > 0:
                prior_mean = float(np.mean(valid_durations))
    
    # 根据数据集类型设置默认先验均值和权重
    if dataset_name == 'Multimodal_SDP':
        default_prior_mean = 140.0
        prior_weight = 0.5
    elif dataset_name == 'INSPIRE':
        default_prior_mean = 120.0
        prior_weight = 0.5
    elif dataset_name in ('MOVER_EPIC', 'MOVER_SIS'):
        default_prior_mean = 120.0
        prior_weight = 0.5
    else:
        default_prior_mean = 108.0
        prior_weight = 0.9
    
    # 如果无法从检索案例获取先验均值，使用默认值
    if prior_mean is None:
        prior_mean = default_prior_mean
    
    # 使用贝叶斯平均计算最终预测值
    if len(all_predictions) >= 1:
        sample_mean = np.mean(all_predictions)
        sample_size = len(all_predictions)
        
        # 贝叶斯更新：posterior_mean = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
        final_predicted_duration = (prior_weight * prior_mean + sample_size * sample_mean) / (prior_weight + sample_size)
        final_temperature = np.mean(all_temperatures)
        final_output = f"多次预测结果: {all_predictions}, 先验均值: {prior_mean:.2f}, 样本均值: {sample_mean:.2f}, 贝叶斯平均结果: {final_predicted_duration:.2f}"
    else:
        # 如果所有预测都失败，使用先验均值
        final_predicted_duration = prior_mean
        final_temperature = 0.0
        final_output = f"预测失败，使用先验均值: {prior_mean:.2f}"
    
    predicted_duration = final_predicted_duration
    predicted_text = final_output
    temperature = final_temperature
    
    # 获取查询案例的基本信息
    query_case_info = get_case_info(row, dataset_name)
    
    # 计算输入大模型的参考案例的手术时长平均值
    # 使用所有检索到的案例（rag_k * 2个）计算统计信息
    reference_duration_avg = None
    reference_case_count = 0
    
    if all_retrieved_cases is not None and len(all_retrieved_cases) > 0:
        # 使用所有检索到的案例（rag_k * 2个）计算均值
        # 提取时长字段
        if dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
            duration_field = 'duration'
        else:
            duration_field = '手术间时长'
        
        if duration_field in all_retrieved_cases.columns:
            durations = all_retrieved_cases[duration_field].values
            valid_durations = durations[~np.isnan(durations)]
            valid_durations = valid_durations[valid_durations > 0]
            if len(valid_durations) > 0:
                reference_duration_avg = float(np.mean(valid_durations))
                reference_case_count = len(valid_durations)
    else:
        # 使用固定参考实例：由于没有fixed_reference_cases_df，无法获取固定参考案例的时长信息
        reference_duration_avg = None
        reference_case_count = 0
    
    # 记录结果
    result_record = {
        'id': row.get('id', '') if dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS') else row.get('手麻系统中手术状态时间历史记录唯一标识', ''),
        'predicted_duration': predicted_duration,
        'ground_true': ground_true,
        'success': predicted_duration is not None,
        'n_predictions': n_predictions,
        'successful_predictions': len(all_predictions),
        'all_predictions': all_predictions,
        'all_temperature': all_temperatures,
        'avg_temperature': np.mean(all_temperatures) if all_temperatures else 0.0,
        'use_rag': use_rag,
        # 查询案例信息
        'query_department': query_case_info['department'],
        'query_surgery': query_case_info['surgery'],
        'query_instrument': query_case_info['instrument'] if query_case_info['instrument'] is not None else None,
        # 参考案例统计信息
        'reference_duration_avg': reference_duration_avg,
        'reference_case_count': reference_case_count
    }
    
    # 如果使用了RAG，添加检索信息
    if use_rag and rag_retriever is not None and rag_mapper is not None:
        try:
            retrieval_results = rag_retriever.retrieve_similar_cases(row, k=rag_k)
            if retrieval_results:
                retrieval_stats = rag_retriever.get_retrieval_stats(retrieval_results)
                result_record['rag_stats'] = retrieval_stats
        except Exception as e:
            result_record['rag_error'] = str(e)
    
    return result_record

def extract_duration_from_text(text):
    """
    从模型输出文本中提取手术时长数值。
    Args:
        text (str): 模型输出的文本
    
    Returns:
        float: 提取的手术时长数值，如果无法提取则返回None
    """
    if not text:
        return None

    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    def _to_positive_float(value_str):
        try:
            duration = float(value_str)
            return duration if duration > 0 else None
        except (ValueError, TypeError):
            return None

    # 1) 纯数字（如 "220.0"）
    duration = _to_positive_float(cleaned_text)
    if duration is not None:
        return duration

    # 2) 冒号后取值：同时支持英文 ':' 与中文 '：'
    #    例: "Surgery duration: 220.0" / "手术时长：220.0"
    last_colon_index = max(cleaned_text.rfind(':'), cleaned_text.rfind('：'))
    if last_colon_index != -1:
        after_colon = cleaned_text[last_colon_index + 1:].strip()
        # 直接可转 float
        duration = _to_positive_float(after_colon)
        if duration is not None:
            return duration
        # 冒号后还带单位/其它文字，如 "220.0 min" / "220 minutes"
        match = re.search(r'([-+]?\d*\.?\d+)', after_colon)
        if match:
            duration = _to_positive_float(match.group(1))
            if duration is not None:
                return duration

    # 3) 兜底：从全文提取最后一个正数（兼容多行/夹杂解释）
    matches = re.findall(r'([-+]?\d*\.?\d+)', cleaned_text)
    for value_str in reversed(matches):
        duration = _to_positive_float(value_str)
        if duration is not None:
            return duration

    return None

def evaluate_multiple_strategies(results: list, dataset_name: str = 'OurSDP') -> None:
    """
    评估多种预测聚合策略的性能
    
    Args:
        results: 预测结果列表
        dataset_name: 数据集名称
    """
    if not results:
        return
    
    # 直接使用 EnsembleStrategies 类，传入内存中的结果列表
    evaluator = EnsembleStrategies(results_file=None, results=results)
    
    # 定义要评估的策略（只评估主要策略）
    strategy_names = [
        'current_method*',  # 当前实现策略（去掉头尾后取平均）
        'simple_average',   # 简单平均
        'trimmed_mean',     # 截尾平均
        'median',           # 中位数
        'robust_mean',      # 鲁棒平均
        'confidence_weighted',  # 置信度加权
        'bayesian_average', # 贝叶斯平均（默认先验）
        'bayesian_average_withRAG',  # 贝叶斯平均（使用RAG先验）
    ]
    
    # 评估指定策略
    strategy_results = {}
    for strategy_name in strategy_names:
        try:
            # 临时禁用策略评估时的打印输出，避免重复信息
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            metrics = evaluator.evaluate_strategy(strategy_name)
            
            # 恢复输出
            sys.stdout = old_stdout
            
            if metrics:
                strategy_results[strategy_name] = metrics
        except Exception as e:
            print(f"策略 {strategy_name} 评估出错: {e}")
    
    # 打印对比结果
    if strategy_results:
        print("\n" + "="*80)
        print("多次预测聚合策略性能对比")
        print("="*80)
        print(f"{'策略名称':<30} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10} {'成功率':<10}")
        print("-"*80)
        
        # 按MAE排序
        sorted_strategies = sorted(strategy_results.items(), key=lambda x: x[1]['MAE'])
        
        for strategy_name, metrics in sorted_strategies:
            print(f"{strategy_name:<30} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f} "
                  f"{metrics['R2']:<10.4f} {metrics['MAPE']:<10.2f} {metrics['success_rate']:<10.2f}")

def dataset_load(root_path=None):
    if root_path is None:
        root_path = "/home/likx/time_series_forecasting/SDP_DATASET"

    train_df = read_dataset(os.path.join(root_path, "train.pkl"))
    val_df = read_dataset(os.path.join(root_path, "val.pkl"))
    test_df = read_dataset(os.path.join(root_path, "test.pkl"))

    return train_df, val_df, test_df

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM4SDP 手术时长预测')
    parser.add_argument('--port', type=int, default=8000, help='API服务器端口号 (默认: 8000)')
    parser.add_argument('--model', type=str, default='llama3', 
                        help='模型名称: 例如 llama3、qwen3、deepseek-r1 (默认: llama3)')
    parser.add_argument('--api_key', type=str, default=None, help='API密钥 (默认: 从环境变量获取)')
    parser.add_argument('--max_workers', type=int, default=4, help='并发线程数 (默认: 4)')
    parser.add_argument('--use_rag', action='store_true', help='是否使用RAG检索增强 (默认: False)')
    parser.add_argument('--rag_k', type=int, default=5, help='RAG检索案例数量 (默认: 5)')
    parser.add_argument('--force_rebuild_rag', action='store_true', help='强制重建RAG数据库 (默认: False)')
    parser.add_argument('--force_reencode', action='store_true', help='强制重新编码特征，忽略缓存 (默认: False)')
    parser.add_argument('--index_type', type=str, default='IVFFlat', help='索引类型: Flat 或 IVFFlat (默认: IVFFlat)')
    parser.add_argument('--enable_prior_hint', action='store_true', help='是否在Prompt中加入层次先验提示 (默认: False)')
    parser.add_argument('--dataset_name', type=str, default='OurSDP', help='数据集名称')
    parser.add_argument('--ran_path', type=str, default=None, help='RAN模型目录路径')
    parser.add_argument('--use_grouped_encoding', action='store_true', help='使用分组编码（按权重分组编码BERT特征，默认: 独立编码）')
    parser.add_argument(
        '--rag_weight_scheme',
        type=str,
        default='mutual_info',
        choices=list(WEIGHT_SCHEMES),
        help='RAG 特征权重方案 (默认: mutual_info)。paper_pca 为论文内置 PCA。',
    )
    parser.add_argument(
        '--rag_weight_json',
        type=str,
        default=None,
        help='可选：直接指定 weights.json，覆盖 scheme 文件',
    )
    args = parser.parse_args()
    set_weight_scheme(args.rag_weight_scheme, weight_json=args.rag_weight_json)

    # 处理RAN模型路径
    if args.ran_path is not None:
        if os.path.isdir(args.ran_path):
            model_name = f"ran_{args.dataset_name or 'OurSDP'}.pt"
            args.ran_path = os.path.join(args.ran_path, model_name)
            print(f"RAN模型路径: {args.ran_path}")
        elif os.path.isfile(args.ran_path):
            print(f"RAN模型路径: {args.ran_path}")
        else:
            print(f"⚠ RAN模型路径不存在: {args.ran_path}，将不使用RAN")

    # 根据是否使用分组编码设置数据库路径
    if args.use_grouped_encoding:
        base_path = "/home/data2/lkx_data/FAISSVectorDatabase_grouped"
    else:
        base_path = "/home/data2/lkx_data/FAISSVectorDatabase"
    if args.dataset_name:
        VectorDatabase_path = f"{base_path}_{args.dataset_name}"
    else:
        VectorDatabase_path = base_path
    VectorDatabase_path = VectorDatabase_path + faiss_weight_suffix(
        args.rag_weight_scheme, args.rag_weight_json
    )
    
    Llama_model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"
    gemma3_model_id = "/home/data2/LLM_benchmarking/gemma-3-4b-it"
    gemma3_12b_model_id = "/home/data2/LLM_benchmarking/gemma-3-12b-it"
    Qwen_model_id = "/home/data2/LLM_benchmarking/Qwen3-8B"
    Qwen_4b_model_id = "/home/data2/models/Qwen3-4B"
    Qwen_14b_model_id = "/home/data/models/Qwen3-14B"
    Qwen_32b_model_id = "/home/data2/LLM_benchmarking/Qwen3-32B"
    DeepSeek_model_id = "/home/data2/LLM_benchmarking/DeepSeek-R1-Distill-Llama-8B"
    HuatuoGPT_model_id = "/home/data2/LLM_benchmarking/HuatuoGPT-o1-7B"
    MedResearch_model_id = "/home/data2/LLM_benchmarking/MedResearcher-R1-32B"
    
    # 根据参数选择模型ID
    if args.model == 'qwen3':
        model_id = Qwen_model_id
    elif args.model == 'qwen3_4b':
        model_id = Qwen_4b_model_id
    elif args.model == 'qwen3_14b':
        model_id = Qwen_14b_model_id
    elif args.model == 'qwen3_32b':
        model_id = Qwen_32b_model_id
    elif args.model == 'deepseek-r1':
        model_id = DeepSeek_model_id
    elif args.model == 'huatuogpt':
        model_id = HuatuoGPT_model_id
    elif args.model == 'gemma3':
        model_id = gemma3_model_id
    elif args.model == 'gemma3_12b':
        model_id = gemma3_12b_model_id
    elif args.model == 'medresearch-r1':
        model_id = MedResearch_model_id
    else:
        model_id = Llama_model_id
    
	# Model Initialization
    USE_EXTERNAL_GPT5 = False  # 改成 True 即可启用外部API
    if USE_EXTERNAL_GPT5:
        # 仅使用外部商业API，不再初始化本地端口的模型
        # gpt-5.2（0.00175), gemini-3-flash-preview-thinking-low (0.0005), glm-5-turbo (0.0182)
        model_id = "glm-5-turbo" 
        client = OpenAI(
            api_key="Your Key",
            base_url="Model URL",
        )
        print(f'【注意】当前正在使用外部商业模型: {model_id}')
    else:
        # 本地部署模型走原来的端口配置
        api_key = args.api_key if args.api_key else os.getenv("API_KEY", "0")
        client = OpenAI(
            api_key=api_key,
            base_url=f"http://localhost:{args.port}/v1",
        )
        print(f"API端口: {args.port}")

    print(f"使用模型: {model_id}")

    # Dataset Loading
    if args.dataset_name == 'Multimodal_SDP':
        train_df, val_df, test_df = dataset_load("/home/likx/time_series_forecasting/SDP_DATASET/Multimodal_SDP")
    elif args.dataset_name == 'INSPIRE':
        train_df, val_df, test_df = dataset_load("/home/likx/time_series_forecasting/SDP_DATASET/INSPIRE")
    elif args.dataset_name == 'MOVER_EPIC':
        train_df, val_df, test_df = dataset_load("/home/likx/time_series_forecasting/SDP_DATASET/MOVER_EPIC")
    elif args.dataset_name == 'MOVER_SIS':
        train_df, val_df, test_df = dataset_load("/home/likx/time_series_forecasting/SDP_DATASET/MOVER_SIS")
    else:
        train_df, val_df, test_df = dataset_load()
    
    # Debug Mode
    # if len(train_df) > 10000:
    #     print(f"调试：将训练集从 {len(train_df)} 条限制为 10000 条")
    #     # train_df = train_df.head(10000)
    #     # 随机选择10000条数据
    #     train_df = train_df.sample(n=10000, random_state=42)
    # 调试：随机选择200条测试集
    # test_df = test_df.sample(n=200, random_state=42)

    # RAG相关初始化
    rag_retriever = None
    rag_mapper = None
    rag_post_processor = None
    prior_hint_generator = None
    
    if args.index_type == 'Flat':
        VectorDatabase_path = VectorDatabase_path + "_Flat"
    elif args.index_type == 'IVFFlat':
        VectorDatabase_path = VectorDatabase_path + "_IVFFlat"
    
    print(f"RAG数据库路径: {VectorDatabase_path}")

    if args.use_rag:
        try:
            # 确保数据库目录存在
            os.makedirs(VectorDatabase_path, exist_ok=True)
            
            # 构建数据库文件路径
            database_path = os.path.join(VectorDatabase_path, "surgery_rag_index.faiss")
            mapping_path = database_path.replace('.faiss', '_mappings.pkl')
            encoder_path = database_path.replace('.faiss', '_encoder.pkl')
            
            # 检查数据库文件是否存在
            database_exists = all(os.path.exists(f) for f in [database_path, mapping_path, encoder_path])
            
            if database_exists and not args.force_rebuild_rag:
                print("加载RAG数据库...")
                try:
                    # 直接加载现有数据库
                    rag_retriever = RAGRetriever(database_path, mapping_path, encoder_path, dataset_name=args.dataset_name, ran_path=args.ran_path)
                    rag_mapper = RAGDataMapper(train_df)
                    
                    if rag_retriever.is_loaded:
                        print("RAG数据库加载成功")
                    else:
                        print("RAG数据库加载失败")
                        raise Exception("数据库加载失败")
                        
                except Exception as e:
                    print(f"加载现有数据库失败: {e}，将重新构建")
                    database_exists = False
            
            if not database_exists or args.force_rebuild_rag:
                print("构建新的RAG数据库...")
                # 构建或加载FAISS数据库
                time_start = time.time()
                rag_database = FAISSVectorDatabase(database_path=database_path, dataset_name=args.dataset_name, use_grouped_encoding=args.use_grouped_encoding)
                success = rag_database.build_database(train_df, force_rebuild=args.force_rebuild_rag, force_reencode=args.force_reencode, index_type=args.index_type, ran_path=args.ran_path)
                build_database_time = time.time() - time_start
                print(f"构建RAG数据库耗时: {build_database_time:.2f} 秒")
                
                if success:
                    rag_database.save_index()
                    
                    # 初始化检索器和映射器
                    rag_retriever = RAGRetriever(database_path, mapping_path, encoder_path, dataset_name=args.dataset_name, ran_path=args.ran_path)
                    rag_mapper = RAGDataMapper(train_df)
                    
                    print("RAG数据库构建成功")
                else:
                    print("RAG数据库构建失败，将使用固定参考实例")
                    args.use_rag = False
                    rag_retriever = None
                    rag_mapper = None
            
            if args.use_rag and rag_retriever and rag_retriever.is_loaded:
                # 初始化后处理器
                # rag_post_processor = RAGPostProcessor(verbose=False, dataset_name=args.dataset_name)
                rag_post_processor = None
                # 若启用先验提示，初始化先验提示生成器
                if args.enable_prior_hint:
                    prior_hint_generator = PriorHintGenerator(dataset_name=args.dataset_name)
                    prior_hint_generator.build_enhanced_priors(train_df)
                    # prior_hint_generator.build_doctor_profiles(train_df)
            else:
                args.use_rag = False
                rag_retriever = None
                rag_mapper = None
                rag_post_processor = None
                prior_hint_generator = None
                
        except Exception as e:
            print(f"RAG检索系统初始化出错: {e}，将使用固定参考实例")
            args.use_rag = False
            rag_retriever = None
            rag_mapper = None
            rag_post_processor = None

    # 构建参考实例
    fixed_reference_instance = fixed_reference_instance_construction(train_df, example_number=args.rag_k, dataset_name=args.dataset_name)
    # fixed_reference_instance = "" # zero_shot_reference_instance    
    # print(f"参考实例: {fixed_reference_instance}")
    # 保存完整prompt示例到文件
    save_prompt_example = True
    if save_prompt_example:
        os.makedirs(f'prompt_logs', exist_ok=True)

    # 存储预测结果
    results = []
    start_time = time.time()
    
    # 模型预测设置
    if USE_EXTERNAL_GPT5:
        n_predictions = 1  # 预测次数
        max_retries_per_prediction = 20  # 最大重试次数        
    else:
        n_predictions = 5  # 预测次数
        max_retries_per_prediction = 10  # 最大重试次数
    
    # 准备所有样本的参数元组
    sample_args = []
    for index, row in test_df.iterrows():
        sample_args.append((
            index, row, client, model_id, args.model, 
            fixed_reference_instance, n_predictions, max_retries_per_prediction, save_prompt_example,
            args.use_rag, rag_retriever, rag_mapper, args.rag_k, rag_post_processor, prior_hint_generator, args.dataset_name
        ))
    
    print(f"使用 {args.max_workers} 个并发预测线程, 总样本数: {len(sample_args)}, 是否使用RAG: {args.use_rag}, 索引类型: {args.index_type}")
    
    # 使用 ThreadPoolExecutor 进行并发预测
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(predict_single_sample, args_tuple): args_tuple[0] 
            for args_tuple in sample_args
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(sample_args), desc="预测进度") as pbar:
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        
                        # 打印结果
                        if result['predicted_duration'] is not None:
                            print(f'样本 {index}: 预测={result["predicted_duration"]:.2f}, 真实={result["ground_true"]}, 成功预测={result["successful_predictions"]}/{n_predictions}')
                        else:
                            print(f'样本 {index}: 预测失败, 真实={result["ground_true"]}, 成功预测={result["successful_predictions"]}/{n_predictions}')
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except Exception as e:
                    print(f'样本 {index} 预测出错: {e}')
                    pbar.update(1)
    
    # 计算总耗时
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / len(results) if results else 0

    # 计算模型性能指标
    metrics = calculate_metrics(results)
    if metrics:
        print("\n" + "="*60)
        print(f"{model_id.split('/')[-1]} 模型性能评估结果")
        print("="*60)
        print_metrics_summary(metrics)
    
    # 评估多种预测聚合策略的性能对比
    evaluate_multiple_strategies(results, dataset_name=args.dataset_name)
    
    # 保存结果到JSON文件
    mode_suffix = "rag" if args.use_rag else "zero_shot" if fixed_reference_instance == "" else "fixed"
    output_file = f'{args.dataset_name}_{mode_suffix}_{args.rag_k}shot_{args.index_type}_{n_predictions}times_{args.model}_prediction_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'总预测 {len(results)} 个样本')
    print(f'总耗时: {total_time:.2f} 秒')
    print(f'平均每个样本耗时: {avg_time_per_sample:.2f} 秒')

if __name__ == "__main__":
    main()
