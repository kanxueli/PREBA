import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class RAGPostProcessor:
    """
    RAG检索后处理器
    负责对检索到的案例进行多层筛选和统计分析，去除噪声案例
    """
    
    def __init__(self, verbose=True, dataset_name=''):
        """
        初始化后处理器
        
        Args:
            verbose: 是否输出详细日志
            dataset_name: 数据集名称，用于选择对应的处理方法
        """
        self.verbose = verbose
        self.dataset_name = dataset_name
        
    # 处理bug
    def process_retrieved_cases(self, query_row: pd.Series, retrieved_cases: pd.DataFrame, 
                              similarity_scores: List[float], target_count: int) -> Tuple[pd.DataFrame, List[float]]:
        """
        对检索到的案例进行分层后处理，筛选出最优质的参考案例
        
        Args:
            query_row: 查询案例
            retrieved_cases: 检索到的案例DataFrame
            similarity_scores: 对应的相似度分数
            target_count: 目标案例数量
        
        Returns:
            tuple: (筛选后的案例DataFrame, 对应的相似度分数)
        """
        if len(retrieved_cases) == 0:
            return pd.DataFrame(), []
        
        # 创建带相似度的案例数据
        cases_with_scores = retrieved_cases.copy()
        cases_with_scores['similarity_score'] = similarity_scores
        
        if self.verbose:
            print(f"    开始分层后处理：原始检索案例数={len(cases_with_scores)}, 目标案例数={target_count}")
        
        # 第一步：科室筛选 - 根据数据集类型选择对应方法
        if self.dataset_name == 'Multimodal_SDP':
            same_department_cases = self._filter_by_department_MMSDP(query_row, cases_with_scores)
        else:
            same_department_cases = self._filter_by_department(query_row, cases_with_scores)
        
        # 判断同科室案例数量
        threshold = int(target_count * 1.5)
        
        if len(same_department_cases) < target_count:
            # 情况(1)：同科室案例不足目标数量，需要从其他科室补充
            if self.verbose:
                print(f"    同科室案例数({len(same_department_cases)}) < 目标数量({target_count})，需要跨科室补充")
            
            # 先选择所有同科室案例
            final_cases = same_department_cases.copy()
            shortage = target_count - len(same_department_cases)
            
            if self.verbose:
                print(f"    已选择所有{len(same_department_cases)}个同科室案例，还需补充{shortage}个案例")
            
            # 从其他科室案例中补充
            other_department_cases = cases_with_scores[~cases_with_scores.index.isin(same_department_cases.index)]
            
            if len(other_department_cases) > 0:
                # 以同科室案例为中心进行筛选
                supplement_cases = self._center_based_filtering(
                    same_department_cases, other_department_cases, shortage
                )
                
                if len(supplement_cases) > 0:
                    final_cases = pd.concat([final_cases, supplement_cases], ignore_index=True)
                    if self.verbose:
                        print(f"    从其他科室补充{len(supplement_cases)}个案例，最终获得{len(final_cases)}个案例")
            
            if self.verbose:
                print(f"    跨科室筛选完成，最终案例数={len(final_cases)}")
        elif len(same_department_cases) < threshold:
            # 情况(2)：同科室案例够目标数量但不足阈值，直接统计筛选
            if self.verbose:
                print(f"    同科室案例数({len(same_department_cases)}) >= 目标数量({target_count})但 < 阈值({threshold})，直接统计筛选")
            
            final_cases = self._statistical_filtering(same_department_cases, target_count)
        else:
            # 情况(2)：同科室案例充足，进行分层处理
            if self.verbose:
                print(f"    同科室案例数({len(same_department_cases)}) >= 阈值({threshold})，进行分层筛选")
            
            # 根据数据集类型选择对应的分层筛选方法
            if self.dataset_name == 'Multimodal_SDP':
                final_cases = self._hierarchical_filtering_MMSDP(query_row, same_department_cases, target_count)
            else:
                final_cases = self._hierarchical_filtering(query_row, same_department_cases, target_count)
 
            # 如果分层筛选后案例数量不足目标值，需要补充案例
            if len(final_cases) < target_count:
                shortage = target_count - len(final_cases)
                if self.verbose:
                    print(f"    分层筛选后案例不足：当前{len(final_cases)}个，目标{target_count}个，缺少{shortage}个")
                
                # 从剩余的同科室案例中按相似度选择1.2倍的缺少数量进行统计筛选
                remaining_cases = same_department_cases[~same_department_cases.index.isin(final_cases.index)]
                if len(remaining_cases) > 0:
                    # 选择更多候选案例进行中心化筛选
                    supplement_count = min(int(shortage * 1.2), len(remaining_cases))
                    remaining_sorted = remaining_cases.sort_values('similarity_score', ascending=False)
                    supplement_candidates = remaining_sorted.head(supplement_count)
                    
                    if self.verbose:
                        print(f"    从剩余{len(remaining_cases)}个案例中选择{supplement_count}个进行中心化筛选")
                    
                    # 以现有案例为中心进行筛选
                    supplement_final = self._center_based_filtering(final_cases, supplement_candidates, shortage)
                    
                    # 合并结果
                    if len(supplement_final) > 0:
                        final_cases = pd.concat([final_cases, supplement_final], ignore_index=True)
                        if self.verbose:
                            print(f"    补充{len(supplement_final)}个案例，最终获得{len(final_cases)}个案例")
        
        # 提取结果
        if len(final_cases) > 0:
            final_similarity_scores = final_cases['similarity_score'].tolist()
            final_cases = final_cases.drop('similarity_score', axis=1)
            
            if self.verbose:
                print(f"    后处理完成：最终案例数={len(final_cases)}")
            
            return final_cases, final_similarity_scores
        else:
            if self.verbose:
                print(f"    后处理完成：筛选后无可用案例，返回空结果")
            return pd.DataFrame(), []


    # def process_retrieved_cases(self, query_row: pd.Series, retrieved_cases: pd.DataFrame, 
    #                           similarity_scores: List[float], target_count: int) -> Tuple[pd.DataFrame, List[float]]:
    #     """
    #     对检索到的案例进行分层后处理，筛选出最优质的参考案例
        
    #     Args:
    #         query_row: 查询案例
    #         retrieved_cases: 检索到的案例DataFrame
    #         similarity_scores: 对应的相似度分数
    #         target_count: 目标案例数量
        
    #     Returns:
    #         tuple: (筛选后的案例DataFrame, 对应的相似度分数)
    #     """
    #     if len(retrieved_cases) == 0:
    #         return pd.DataFrame(), []
        
    #     # 创建带相似度的案例数据
    #     cases_with_scores = retrieved_cases.copy()
    #     cases_with_scores['similarity_score'] = similarity_scores
        
    #     if self.verbose:
    #         print(f"    开始分层后处理：原始检索案例数={len(cases_with_scores)}, 目标案例数={target_count}")
        
    #     # 第一步：科室筛选
    #     same_department_cases = self._filter_by_department(query_row, cases_with_scores)
        
    #     # 判断同科室案例数量
    #     threshold = int(target_count * 1.5)
        
    #     if len(same_department_cases) < threshold:
    #         # 情况(1)：同科室案例不足，直接进行统计分析
    #         if self.verbose:
    #             print(f"    同科室案例数({len(same_department_cases)}) < 阈值({threshold})，直接统计筛选")
            
    #         final_cases = self._statistical_filtering(same_department_cases, target_count)
    #     else:
    #         # 情况(2)：同科室案例充足，进行分层处理
    #         if self.verbose:
    #             print(f"    同科室案例数({len(same_department_cases)}) >= 阈值({threshold})，进行分层筛选")
            
    #         final_cases = self._hierarchical_filtering(query_row, same_department_cases, target_count)
 
    #         # 如果分层筛选后案例数量不足目标值，需要补充案例
    #         if len(final_cases) < target_count:
    #             shortage = target_count - len(final_cases)
    #             if self.verbose:
    #                 print(f"    分层筛选后案例不足：当前{len(final_cases)}个，目标{target_count}个，缺少{shortage}个")
                
    #             # 从剩余的同科室案例中按相似度选择1.2倍的缺少数量进行统计筛选
    #             remaining_cases = same_department_cases[~same_department_cases.index.isin(final_cases.index)]
    #             if len(remaining_cases) > 0:
    #                 supplement_count = int(shortage * 1.2)
    #                 supplement_count = min(supplement_count, len(remaining_cases))
                    
    #                 # 按相似度排序选择补充案例
    #                 remaining_sorted = remaining_cases.sort_values('similarity_score', ascending=False)
    #                 supplement_candidates = remaining_sorted.head(supplement_count)
                    
    #                 if self.verbose:
    #                     print(f"    从剩余{len(remaining_cases)}个案例中选择{supplement_count}个进行统计筛选")
                    
    #                 # 对补充候选案例进行统计筛选
    #                 supplement_final = self._statistical_filtering(supplement_candidates, shortage)
                    
    #                 # 合并结果
    #                 if len(supplement_final) > 0:
    #                     final_cases = pd.concat([final_cases, supplement_final], ignore_index=True)
    #                     if self.verbose:
    #                         print(f"    补充{len(supplement_final)}个案例，最终获得{len(final_cases)}个案例")
        
    #     # 提取结果
    #     if len(final_cases) > 0:
    #         final_similarity_scores = final_cases['similarity_score'].tolist()
    #         final_cases = final_cases.drop('similarity_score', axis=1)
            
    #         if self.verbose:
    #             print(f"    后处理完成：最终案例数={len(final_cases)}")
            
    #         return final_cases, final_similarity_scores
    #     else:
    #         if self.verbose:
    #             print(f"    后处理完成：筛选后无可用案例，返回空结果")
    #         return pd.DataFrame(), []
    
    def _filter_by_department(self, query_row: pd.Series, cases_with_scores: pd.DataFrame) -> pd.DataFrame:
        """
        第一步：按科室筛选，只保留同科室的案例
        """
        query_department = query_row.get('患者住院科室')
        
        # 统一数据类型：将案例数据转换为与查询数据相同的类型
        if isinstance(query_department, str):
            cases_department = cases_with_scores['患者住院科室'].astype(str)
        else:
            cases_department = cases_with_scores['患者住院科室']
        
        # 筛选同科室案例
        same_department_mask = cases_department == query_department
        filtered_cases = cases_with_scores[same_department_mask].copy()
        
        if self.verbose:
            print(f"      科室筛选：查询科室={query_department}, 同科室案例数={len(filtered_cases)}")
        
        return filtered_cases
    
    def _hierarchical_filtering(self, query_row: pd.Series, same_department_cases: pd.DataFrame, 
                               target_count: int) -> pd.DataFrame:
        """
        分层筛选：基于手术特征的逐层匹配
        """
        # 获取查询案例的三个关键特征
        query_surgery_name = query_row.get('拟行手术名称')
        query_instrument_name = query_row.get('手术器械清点单手术名称')
        query_diagnosis = query_row.get('主要诊断')
        
        # 统一数据类型
        if isinstance(query_surgery_name, str):
            cases_surgery_name = same_department_cases['拟行手术名称'].astype(str)
        else:
            cases_surgery_name = same_department_cases['拟行手术名称']
            
        if isinstance(query_instrument_name, str):
            cases_instrument_name = same_department_cases['手术器械清点单手术名称'].astype(str)
        else:
            cases_instrument_name = same_department_cases['手术器械清点单手术名称']
            
        if isinstance(query_diagnosis, str):
            cases_diagnosis = same_department_cases['主要诊断'].astype(str)
        else:
            cases_diagnosis = same_department_cases['主要诊断']
        
        candidate_list = pd.DataFrame()
        
        if self.verbose:
            print(f"      分层筛选：拟行手术='{query_surgery_name}', 器械清点='{query_instrument_name}', 主要诊断='{query_diagnosis}'")
        
        # 第一层：三个特征完全匹配
        layer1_mask = (
            (cases_surgery_name == query_surgery_name) &
            (cases_instrument_name == query_instrument_name) &
            (cases_diagnosis == query_diagnosis)
        )
        layer1_cases = same_department_cases[layer1_mask]
        
        if len(layer1_cases) > 0:
            candidate_list = pd.concat([candidate_list, layer1_cases], ignore_index=True)
            if self.verbose:
                print(f"        第一层(3特征匹配)：{len(layer1_cases)}个案例")
        
        # 检查是否已达到目标数量
        if len(candidate_list) == target_count:
            return candidate_list
        elif len(candidate_list) > target_count:
            return self._statistical_filtering(candidate_list, target_count)
        
        # 第二层：任意两个特征匹配
        if len(candidate_list) < target_count:
            layer2_mask = (
                ((cases_surgery_name == query_surgery_name) & (cases_instrument_name == query_instrument_name)) |
                ((cases_surgery_name == query_surgery_name) & (cases_diagnosis == query_diagnosis)) |
                ((cases_instrument_name == query_instrument_name) & (cases_diagnosis == query_diagnosis))
            ) & (~layer1_mask)  # 排除第一层已选中的案例
            
            layer2_cases = same_department_cases[layer2_mask]
            
            if len(layer2_cases) > 0:
                if self.verbose:
                    print(f"        第二层(2特征匹配)：{len(layer2_cases)}个案例")
                
                # 如果第一层+第二层刚好等于目标数量
                if len(candidate_list) + len(layer2_cases) == target_count:
                    candidate_list = pd.concat([candidate_list, layer2_cases], ignore_index=True)
                    return candidate_list
                elif len(candidate_list) + len(layer2_cases) > target_count:
                    # 需要进行以现有案例为中心的筛选
                    needed_count = target_count - len(candidate_list)
                    selected_layer2 = self._center_based_filtering(candidate_list, layer2_cases, needed_count)
                    candidate_list = pd.concat([candidate_list, selected_layer2], ignore_index=True)
                    return candidate_list
                else:
                    # 第二层案例不够，全部加入
                    candidate_list = pd.concat([candidate_list, layer2_cases], ignore_index=True)
        
        # 第三层：任意一个特征匹配
        if len(candidate_list) < target_count:
            layer3_mask = (
                (cases_surgery_name == query_surgery_name) |
                (cases_instrument_name == query_instrument_name) |
                (cases_diagnosis == query_diagnosis)
            ) & (~layer1_mask) & (~layer2_mask)  # 排除前两层已选中的案例
            
            layer3_cases = same_department_cases[layer3_mask]
            
            if len(layer3_cases) > 0:
                if self.verbose:
                    print(f"        第三层(1特征匹配)：{len(layer3_cases)}个案例")
                
                needed_count = target_count - len(candidate_list)
                if len(layer3_cases) <= needed_count:
                    # 第三层案例不够，全部加入
                    candidate_list = pd.concat([candidate_list, layer3_cases], ignore_index=True)
                else:
                    # 进行以现有案例为中心的筛选
                    selected_layer3 = self._center_based_filtering(candidate_list, layer3_cases, needed_count)
                    candidate_list = pd.concat([candidate_list, selected_layer3], ignore_index=True)
        
        return candidate_list
    
    def _filter_by_department_MMSDP(self, query_row: pd.Series, cases_with_scores: pd.DataFrame) -> pd.DataFrame:
        """
        MMSDP数据集：按科室筛选，只保留同科室的案例
        """
        # 如果字段不存在，跳过筛选
        if 'department' not in cases_with_scores.columns:
            if self.verbose:
                print(f"      警告：字段 'department' 不存在，跳过科室筛选")
            return cases_with_scores
        
        # 获取查询科室值，确保是标量
        def extract_scalar(value):
            """从可能为列表/数组的值中提取标量"""
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                if len(value) > 0:
                    return value[0] if not pd.isna(value[0]) else ''
                return ''
            return value
        
        query_department = extract_scalar(query_row.get('department'))
        query_department = str(query_department) if query_department is not None and not pd.isna(query_department) else ''
        
        # 统一数据类型：将案例数据转换为字符串
        cases_department = cases_with_scores['department'].astype(str)
        
        # 筛选同科室案例
        same_department_mask = cases_department == query_department
        filtered_cases = cases_with_scores[same_department_mask].copy()
        
        if self.verbose:
            print(f"      科室筛选：查询科室={query_department}, 同科室案例数={len(filtered_cases)}")
        
        return filtered_cases
    
    def _hierarchical_filtering_MMSDP(self, query_row: pd.Series, same_department_cases: pd.DataFrame, 
                                      target_count: int) -> pd.DataFrame:
        """
        MMSDP数据集：分层筛选 - 基于手术特征的逐层匹配
        使用三个特征：name（手术名称）、patient_diagnose（主要诊断）、level（手术等级）
        """
        # 获取查询案例的三个关键特征，并确保是标量值
        def extract_scalar(value):
            """从可能为列表/数组的值中提取标量"""
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                if len(value) > 0:
                    return value[0] if not pd.isna(value[0]) else ''
                return ''
            return value
        
        query_surgery_name = extract_scalar(query_row.get('name'))
        query_diagnosis = extract_scalar(query_row.get('patient_diagnose'))
        query_level = extract_scalar(query_row.get('level'))
        
        # 统一数据类型：将查询值转换为字符串，案例数据也转换为字符串
        query_surgery_name = str(query_surgery_name) if query_surgery_name is not None and not pd.isna(query_surgery_name) else ''
        query_diagnosis = str(query_diagnosis) if query_diagnosis is not None and not pd.isna(query_diagnosis) else ''
        query_level = str(query_level) if query_level is not None and not pd.isna(query_level) else ''
        
        cases_surgery_name = same_department_cases['name'].astype(str)
        cases_diagnosis = same_department_cases['patient_diagnose'].astype(str)
        cases_level = same_department_cases['level'].astype(str)
        
        candidate_list = pd.DataFrame()
        
        if self.verbose:
            print(f"      分层筛选：拟行手术='{query_surgery_name}', 主要诊断='{query_diagnosis}', 手术等级='{query_level}'")
        
        # 第一层：三个特征完全匹配
        layer1_mask = (
            (cases_surgery_name == query_surgery_name) &
            (cases_diagnosis == query_diagnosis) &
            (cases_level == query_level)
        )
        layer1_cases = same_department_cases[layer1_mask]
        
        if len(layer1_cases) > 0:
            candidate_list = pd.concat([candidate_list, layer1_cases], ignore_index=True)
            if self.verbose:
                print(f"        第一层(3特征匹配)：{len(layer1_cases)}个案例")
        
        # 检查是否已达到目标数量
        if len(candidate_list) == target_count:
            return candidate_list
        elif len(candidate_list) > target_count:
            return self._statistical_filtering(candidate_list, target_count)
        
        # 第二层：任意两个特征匹配
        if len(candidate_list) < target_count:
            layer2_mask = (
                ((cases_surgery_name == query_surgery_name) & (cases_diagnosis == query_diagnosis)) |
                ((cases_surgery_name == query_surgery_name) & (cases_level == query_level)) |
                ((cases_diagnosis == query_diagnosis) & (cases_level == query_level))
            ) & (~layer1_mask)  # 排除第一层已选中的案例
            
            layer2_cases = same_department_cases[layer2_mask]
            
            if len(layer2_cases) > 0:
                if self.verbose:
                    print(f"        第二层(2特征匹配)：{len(layer2_cases)}个案例")
                
                # 如果第一层+第二层刚好等于目标数量
                if len(candidate_list) + len(layer2_cases) == target_count:
                    candidate_list = pd.concat([candidate_list, layer2_cases], ignore_index=True)
                    return candidate_list
                elif len(candidate_list) + len(layer2_cases) > target_count:
                    # 需要进行以现有案例为中心的筛选
                    needed_count = target_count - len(candidate_list)
                    selected_layer2 = self._center_based_filtering(candidate_list, layer2_cases, needed_count)
                    candidate_list = pd.concat([candidate_list, selected_layer2], ignore_index=True)
                    return candidate_list
                else:
                    # 第二层案例不够，全部加入
                    candidate_list = pd.concat([candidate_list, layer2_cases], ignore_index=True)
        
        # 第三层：任意一个特征匹配
        if len(candidate_list) < target_count:
            layer3_mask = (
                (cases_surgery_name == query_surgery_name) |
                (cases_diagnosis == query_diagnosis) |
                (cases_level == query_level)
            ) & (~layer1_mask) & (~layer2_mask)  # 排除前两层已选中的案例
            
            layer3_cases = same_department_cases[layer3_mask]
            
            if len(layer3_cases) > 0:
                if self.verbose:
                    print(f"        第三层(1特征匹配)：{len(layer3_cases)}个案例")
                
                needed_count = target_count - len(candidate_list)
                if len(layer3_cases) <= needed_count:
                    # 第三层案例不够，全部加入
                    candidate_list = pd.concat([candidate_list, layer3_cases], ignore_index=True)
                else:
                    # 进行以现有案例为中心的筛选
                    selected_layer3 = self._center_based_filtering(candidate_list, layer3_cases, needed_count)
                    candidate_list = pd.concat([candidate_list, selected_layer3], ignore_index=True)
        
        return candidate_list
    
    def _center_based_filtering(self, center_cases: pd.DataFrame, candidate_cases: pd.DataFrame, 
                               needed_count: int) -> pd.DataFrame:
        """
        以现有案例为中心的筛选：选择与已有案例时长最接近的案例
        """
        if len(center_cases) == 0 or len(candidate_cases) == 0:
            return candidate_cases.head(needed_count)
        
        # 根据数据集类型选择时长字段名
        duration_field = 'duration' if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS') else '手术间时长'
        
        # 计算中心案例的时长统计
        center_durations = []
        for _, row in center_cases.iterrows():
            duration_str = self._safe_get_value(row, duration_field)
            try:
                duration = float(duration_str)
                if duration > 0:
                    center_durations.append(duration)
            except (ValueError, TypeError):
                continue
        
        if not center_durations:
            return candidate_cases.head(needed_count)
        
        center_mean = np.mean(center_durations)
        center_std = np.std(center_durations) if len(center_durations) > 1 else 0
        
        # 计算候选案例与中心的距离分数
        candidate_scores = []
        for idx, row in candidate_cases.iterrows():
            duration_str = self._safe_get_value(row, duration_field)
            try:
                duration = float(duration_str)
                if duration > 0:
                    # 计算与中心的距离分数（越小越好）
                    if center_std > 0:
                        distance_score = abs(duration - center_mean) / center_std
                    else:
                        distance_score = abs(duration - center_mean)
                else:
                    distance_score = float('inf')
            except (ValueError, TypeError):
                distance_score = float('inf')
            
            # 结合相似度分数（越大越好）
            similarity_score = row.get('similarity_score', 0)
            # 综合分数：相似度权重0.7，距离权重0.3
            combined_score = similarity_score * 0.7 - distance_score * 0.3
            candidate_scores.append((combined_score, idx))
        
        # 按综合分数排序，选择top-k
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in candidate_scores[:needed_count]]
        
        return candidate_cases.loc[selected_indices]
    
    
    def _statistical_filtering(self, cases_with_scores: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        第三步：统计分析和离群值处理，去除手术时长异常的案例
        """
        if len(cases_with_scores) <= target_count:
            if self.verbose:
                print(f"      统计筛选：案例数不足目标数量，直接返回所有案例")
            return cases_with_scores
        
        # 根据数据集类型选择时长字段名
        duration_field = 'duration' if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS') else '手术间时长'
        
        # 获取手术时长数据
        durations = []
        valid_indices = []
        
        for idx, row in cases_with_scores.iterrows():
            duration = self._safe_get_value(row, duration_field)
            try:
                duration_float = float(duration)
                if duration_float > 0:  # 只考虑正数时长
                    durations.append(duration_float)
                    valid_indices.append(idx)
            except (ValueError, TypeError):
                continue
        
        if len(durations) <= target_count:
            if self.verbose:
                print(f"      统计筛选：有效时长案例数不足，返回所有有效案例")
            return cases_with_scores.loc[valid_indices]
        
        durations = np.array(durations)
        
        # 计算统计指标
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        std_duration = np.std(durations)
        q1 = np.percentile(durations, 25)
        q3 = np.percentile(durations, 75)
        iqr = q3 - q1
        
        if self.verbose:
            print(f"      统计筛选：时长统计 - 均值={mean_duration:.1f}, 中位数={median_duration:.1f}, 标准差={std_duration:.1f}")
            print(f"        四分位数：Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
        
        # 使用多种方法识别离群值并综合判断
        outlier_scores = self._calculate_outlier_scores(durations, mean_duration, median_duration, 
                                                       std_duration, q1, q3, iqr)
        
        # 为每个案例计算综合质量分数（相似度 + 时长合理性）
        quality_scores = []
        for i, idx in enumerate(valid_indices):
            similarity = cases_with_scores.loc[idx, 'similarity_score']
            outlier_penalty = outlier_scores[i]
            
            # 综合分数 = 相似度分数 * (1 - 离群惩罚)
            quality_score = similarity * (1 - outlier_penalty)
            quality_scores.append((quality_score, idx))
        
        # 按质量分数排序并选择top-k
        quality_scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in quality_scores[:target_count]]
        
        final_cases = cases_with_scores.loc[selected_indices]
        
        if self.verbose:
            # 根据数据集类型选择时长字段名
            duration_field = 'duration' if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS') else '手术间时长'
            selected_durations = [float(self._safe_get_value(final_cases.loc[idx], duration_field)) 
                                for idx in selected_indices]
            print(f"      统计筛选：最终选择{len(final_cases)}个案例，时长范围={min(selected_durations):.1f}-{max(selected_durations):.1f}")
        
        return final_cases
    
    def _calculate_outlier_scores(self, durations: np.ndarray, mean_val: float, median_val: float,
                                 std_val: float, q1: float, q3: float, iqr: float) -> List[float]:
        """
        计算每个案例的离群值分数（0-1之间，越高表示越可能是离群值）
        """
        outlier_scores = []
        
        for duration in durations:
            score = 0.0
            
            # 方法1：Z-score方法（基于标准差）
            if std_val > 0:
                z_score = abs(duration - mean_val) / std_val
                if z_score > 2.0:  # 超过2个标准差
                    score += min(0.4, (z_score - 2.0) * 0.1)
            
            # 方法2：IQR方法（基于四分位距）
            if iqr > 0:
                if duration < q1 - 1.5 * iqr or duration > q3 + 1.5 * iqr:
                    # 计算超出合理范围的程度
                    if duration < q1 - 1.5 * iqr:
                        excess = (q1 - 1.5 * iqr - duration) / iqr
                    else:
                        excess = (duration - q3 - 1.5 * iqr) / iqr
                    score += min(0.4, excess * 0.1)
            
            # 方法3：基于中位数的偏差
            if median_val > 0:
                relative_deviation = abs(duration - median_val) / median_val
                if relative_deviation > 0.5:  # 偏离中位数超过50%
                    score += min(0.2, (relative_deviation - 0.5) * 0.2)
            
            outlier_scores.append(min(1.0, score))  # 确保分数不超过1
        
        return outlier_scores
    
    def _safe_get_value(self, row: pd.Series, key: str, default='') -> str:
        """安全获取数据值，处理空值"""
        value = row.get(key, default)
        
        if value is None or pd.isna(value):
            return '无'
        
        str_value = str(value).strip()
        if str_value == '' or str_value == 'nan' or str_value == 'None':
            return '无'
        
        return str_value
    
    def get_filtering_stats(self, original_count: int, final_count: int, 
                          department_matches: int, surgery_matches: int) -> Dict:
        """
        获取筛选统计信息
        """
        return {
            'original_count': original_count,
            'department_filtered_count': department_matches,
            'surgery_filtered_count': surgery_matches,
            'final_count': final_count,
            'filter_ratio': final_count / original_count if original_count > 0 else 0
        }
