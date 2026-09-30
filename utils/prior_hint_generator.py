#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
先验提示生成器
负责生成丰富的先验统计信息和医学知识提示
"""

import pandas as pd
import numpy as np
import pickle
import os
from .prompt_constructor import hospital_departments
from typing import Dict, Tuple, Optional

class PriorHintGenerator:
    """先验提示生成器"""
    
    def __init__(self, dataset_name=''):
        """
        初始化先验提示生成器
        
        Args:
            dataset_name: 数据集名称，用于选择字段名 ('Multimodal_SDP' 或其他)
        """
        self.dataset_name = dataset_name
        self.priors = {}
        self.medical_knowledge = self._init_medical_knowledge()
        self.doctor_profiles = {}
    
    def _init_medical_knowledge(self) -> Dict:
        """初始化医学知识库"""
        return {
            'surgery_complexity': {
                '一类': {'base_time': 30, 'complexity': '简单'},
                '二类': {'base_time': 60, 'complexity': '中等'},
                '三类': {'base_time': 120, 'complexity': '复杂'},
                '四类': {'base_time': 180, 'complexity': '高复杂'}
            },
            'asa_risk_factors': {
                'Ⅰ 级': {'risk_multiplier': 1.0, 'description': '健康患者'},
                'Ⅱ 级': {'risk_multiplier': 1.1, 'description': '轻度系统性疾病'},
                'Ⅲ 级': {'risk_multiplier': 1.3, 'description': '重度系统性疾病'},
                'Ⅳ 级': {'risk_multiplier': 1.5, 'description': '威胁生命的系统性疾病'},
                'Ⅴ 级': {'risk_multiplier': 1.8, 'description': '濒死患者'}
            },
            'age_risk_factors': {
                'young': {'age_range': (0, 18), 'risk_multiplier': 0.9, 'description': '年轻患者'},
                'adult': {'age_range': (18, 65), 'risk_multiplier': 1.0, 'description': '成年患者'},
                'elderly': {'age_range': (65, 80), 'risk_multiplier': 1.2, 'description': '老年患者'},
                'very_elderly': {'age_range': (80, 120), 'risk_multiplier': 1.4, 'description': '高龄患者'}
            }
        }
    
    def build_enhanced_priors(self, df: pd.DataFrame) -> Dict:
        """构建增强的先验统计信息"""
        priors = {}
        if df is None or len(df) == 0:
            return priors
        
        print("开始构建先验统计信息...")
        
        tmp = df.copy()
        print(f"原始数据形状: {tmp.shape}")
        
        # 根据数据集类型选择字段名
        if self.dataset_name == 'Multimodal_SDP':
            duration_field = 'duration'
            group_keys_levels = [
                ['department', 'procedures', 'level'],
                ['department', 'procedures'],
                ['department']
            ]
        elif self.dataset_name == 'INSPIRE':
            duration_field = 'duration'
            group_keys_levels = [
                ['department.1', 'ICD'],
                ['department.1']
            ]
        elif self.dataset_name == 'MOVER_EPIC':
            duration_field = 'duration'
            group_keys_levels = [
                ['procedure_name', 'diagnosis_names'],
                ['procedure_name']
            ]
        elif self.dataset_name == 'MOVER_SIS':
            duration_field = 'duration'
            group_keys_levels = [
                ['procedure_name', 'preop_medications'],
                ['procedure_name']
            ]
        else:
            duration_field = '手术间时长'
            group_keys_levels = [
                ['患者住院科室','拟行手术名称','手术器械清点单手术名称','手术等级'],
                ['患者住院科室','拟行手术名称','手术器械清点单手术名称'],
                ['患者住院科室','拟行手术名称'],
                ['患者住院科室','手术器械清点单手术名称'],
                ['患者住院科室']
            ]
        
        # 安全处理手术时长列
        try:
            tmp['__dur__'] = tmp[duration_field].apply(self._safe_to_float)
            print(f"手术时长列处理完成，数据类型: {tmp['__dur__'].dtype}")
        except Exception as e:
            print(f"处理手术时长列时出错: {e}")
            return priors
        
        # 过滤有效的手术时长
        print(f"过滤前数据量: {len(tmp)}")
        tmp = tmp[tmp['__dur__'] > 0]
        print(f"过滤后数据量: {len(tmp)}")
        
        for keys in group_keys_levels:
            try:
                g = tmp.groupby(keys)['__dur__'].agg(['median', 'mean', 'count', 'std', 'min', 'max'])
                g['q25'] = tmp.groupby(keys)['__dur__'].quantile(0.25)
                g['q75'] = tmp.groupby(keys)['__dur__'].quantile(0.75)
                
                for idx, row in g.iterrows():
                    if not isinstance(idx, tuple):
                        idx = (idx,)
                    priors[(len(keys),) + tuple(idx)] = {
                        'median': float(row['median']),
                        'mean': float(row['mean']),
                        'count': int(row['count']),
                        'std': float(row['std']) if not pd.isna(row['std']) else 0.0,
                        'q25': float(row['q25']) if not pd.isna(row['q25']) else 0.0,
                        'q75': float(row['q75']) if not pd.isna(row['q75']) else 0.0,
                        'min': float(row['min']) if not pd.isna(row['min']) else 0.0,
                        'max': float(row['max']) if not pd.isna(row['max']) else 0.0
                    }
            except Exception:
                continue
        
        self.priors = priors
        return priors
    
    def build_doctor_profiles(self, df: pd.DataFrame) -> Dict:
        # 首先尝试加载已保存的医生画像
        if self._load_doctor_profiles():
            print(f"加载已保存的医生先验统计，共{len(self.doctor_profiles)}位医生")
            return self.doctor_profiles
        
        # 如果没有保存的医生画像，则重新构建
        print("开始构建医生先验统计")
        doctor_profiles = {}
        if df is None or len(df) == 0:
            return doctor_profiles
        
        # 处理所有可能的医生ID
        all_surgeon_ids = set()
        for surgeon_id in df['手术医生']:
            if isinstance(surgeon_id, (list, tuple)):
                for single_id in surgeon_id:
                    if single_id != '无' and not pd.isna(single_id) and single_id != 0:
                        all_surgeon_ids.add(str(single_id))
            else:
                if surgeon_id != '无' and not pd.isna(surgeon_id) and surgeon_id != 0:
                    all_surgeon_ids.add(str(surgeon_id))
    
        # 为每位医生构建画像
        for surgeon_id in all_surgeon_ids:
            # 获取该医生的所有案例
            surgeon_cases = df[df['手术医生'].apply(lambda x: self._contains_doctor(x, surgeon_id))]
            if len(surgeon_cases) < 3:
                continue
            
            surgeon_priors = self._build_surgeon_priors(surgeon_cases)
            if surgeon_priors:
                # 计算该医生的总体时长统计
                durations = []
                for _, case in surgeon_cases.iterrows():
                    duration = self._safe_to_float(case.get('手术间时长', 0))
                    if duration > 0:
                        durations.append(duration)
                
                overall_stats = {}
                if durations:
                    import numpy as np
                    durations = np.array(durations)
                    overall_stats = {
                        'median': float(np.median(durations)),
                        'mean': float(np.mean(durations)),
                        'std': float(np.std(durations)),
                        'min': float(np.min(durations)),
                        'max': float(np.max(durations)),
                        'q25': float(np.percentile(durations, 25)),
                        'q75': float(np.percentile(durations, 75)),
                        'count': len(durations)
                    }
                
                doctor_profiles[f"surgeon_{surgeon_id}"] = {
                    'type': 'surgeon',
                    'priors': surgeon_priors,
                    'total_cases': len(surgeon_cases),
                    'overall_stats': overall_stats
                }
        
        self.doctor_profiles = doctor_profiles
        print(f"医生先验统计构建完成，共{len(doctor_profiles)}位医生")
        
        self._save_doctor_profiles()
        
        return doctor_profiles
    
    def _build_surgeon_priors(self, surgeon_cases: pd.DataFrame) -> Dict:
        priors = {}
        if len(surgeon_cases) == 0:
            return priors
        
        tmp = surgeon_cases.copy()
        
        # 安全处理手术时长列
        try:
            tmp['__dur__'] = tmp['手术间时长'].apply(self._safe_to_float)
        except Exception:
            return priors
        
        # 过滤有效的手术时长
        tmp = tmp[tmp['__dur__'] > 0]
        if len(tmp) == 0:
            return priors
        
        # 使用与build_enhanced_priors完全相同的层次
        group_keys_levels = [
            ['拟行手术名称','手术器械清点单手术名称','手术等级'],
            ['拟行手术名称','手术器械清点单手术名称'],
            ['拟行手术名称'],
            ['手术器械清点单手术名称'],
            ['患者住院科室','手术等级']
        ]   
        
        for keys in group_keys_levels:
            try:
                g = tmp.groupby(keys)['__dur__'].agg(['median', 'mean', 'count', 'std', 'min', 'max'])
                g['q25'] = tmp.groupby(keys)['__dur__'].quantile(0.25)
                g['q75'] = tmp.groupby(keys)['__dur__'].quantile(0.75)
                
                for idx, row in g.iterrows():
                    if not isinstance(idx, tuple):
                        idx = (idx,)
                    priors[(len(keys),) + tuple(idx)] = {
                        'median': float(row['median']),
                        'mean': float(row['mean']),
                        'count': int(row['count']),
                        'std': float(row['std']) if not pd.isna(row['std']) else 0.0,
                        'q25': float(row['q25']) if not pd.isna(row['q25']) else 0.0,
                        'q75': float(row['q75']) if not pd.isna(row['q75']) else 0.0,
                        'min': float(row['min']) if not pd.isna(row['min']) else 0.0,
                        'max': float(row['max']) if not pd.isna(row['max']) else 0.0
                    }
            except Exception:
                continue
        
        return priors
    
    def _save_doctor_profiles(self):
        save_dir = "doctor_profiles_cache"
        os.makedirs(save_dir, exist_ok=True)
        
        # 直接保存为PKL格式，保持原始数据结构
        pkl_file = os.path.join(save_dir, "doctor_profiles.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.doctor_profiles, f)
        
        print(f"医生画像已保存: {pkl_file}")
    
    def _load_doctor_profiles(self):
        """加载医生画像（用于缓存）"""
        
        profile_file = "doctor_profiles_cache/doctor_profiles.pkl"
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'rb') as f:
                    self.doctor_profiles = pickle.load(f)
                return True
            except Exception as e:
                print(f"加载医生画像失败: {e}")
                return False
        return False
    
    def query_prior_stats(self, row: pd.Series) -> Tuple[bool, Dict, str]:
        """查询先验统计信息"""
        if not self.priors:
            return False, {}, ""
        
        def extract_scalar(value):
            """从可能为列表/数组的值中提取标量"""
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                if len(value) > 0:
                    return value[0] if not pd.isna(value[0]) else ''
                return ''
            return value
        
        v = lambda k: extract_scalar(row.get(k, ''))
        
        if self.dataset_name == 'Multimodal_SDP':
            candidates = [
                (3, ['department', 'procedures', 'level'], 
                 (v('department'), v('procedures'), v('level')),
                 "住院科室/拟行手术名称/手术等级"),
                (2, ['department', 'procedures'], 
                 (v('department'), v('procedures')),
                 "住院科室/拟行手术名称"),
                (1, ['department'], 
                 (v('department'),),
                 "住院科室")
            ]
        elif self.dataset_name == 'INSPIRE':
            candidates = [
                (2, ['department.1', 'ICD'], (v('department.1'), v('ICD')), "科室/ICD"),
                (1, ['department.1'], (v('department.1'),), "科室")
            ]
        elif self.dataset_name == 'MOVER_EPIC':
            candidates = [
                (2, ['procedure_name', 'diagnosis_names'], (v('procedure_name'), v('diagnosis_names')), "procedure/diagnosis"),
                (1, ['procedure_name'], (v('procedure_name'),), "procedure")
            ]
        elif self.dataset_name == 'MOVER_SIS':
            candidates = [
                (2, ['procedure_name', 'preop_medications'], (v('procedure_name'), v('preop_medications')), "procedure/preop_meds"),
                (1, ['procedure_name'], (v('procedure_name'),), "procedure")
            ]
        else:
            candidates = [
                (4, ['患者住院科室','拟行手术名称','手术器械清点单手术名称','手术等级'], 
                 (v('患者住院科室'), v('拟行手术名称'), v('手术器械清点单手术名称'), v('手术等级')),
                 "患者住院科室/拟行手术名称/手术器械清点单手术名称/手术等级"),
                (3, ['患者住院科室','拟行手术名称','手术器械清点单手术名称'], 
                 (v('患者住院科室'), v('拟行手术名称'), v('手术器械清点单手术名称')),
                 "患者住院科室/拟行手术名称/手术器械清点单手术名称"),
                (2, ['患者住院科室','拟行手术名称'], 
                 (v('患者住院科室'), v('拟行手术名称')),
                 "患者住院科室/拟行手术名称"),
                (2, ['患者住院科室','手术器械清点单手术名称'], 
                 (v('患者住院科室'), v('手术器械清点单手术名称')),
                 "患者住院科室/手术器械清点单手术名称"),
                (1, ['患者住院科室'], 
                 (v('患者住院科室'),),
                 "患者住院科室")
            ]
        
        for lvl, key_names, key_tuple, key_desc in candidates:
            key = (lvl,) + key_tuple
            if key in self.priors:
                return True, self.priors[key], key_desc
        
        return False, {}, ""
    
    def generate_medical_context(self, row: pd.Series) -> str:
        """生成医学上下文信息"""
        context_parts = []
        
        def extract_scalar(value):
            """从可能为列表/数组的值中提取标量"""
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                if len(value) > 0:
                    return value[0] if not pd.isna(value[0]) else ''
                return ''
            return value
        
        # 根据数据集类型选择字段名
        if self.dataset_name == 'Multimodal_SDP':
            surgery_level_field = 'level'
            asa_grade_field = 'assessment_asa'
            age_field = 'patient_age'
        elif self.dataset_name == 'INSPIRE':
            surgery_level_field = None
            asa_grade_field = 'assessment_asa'
            age_field = 'patient_age'
        elif self.dataset_name == 'MOVER_EPIC':
            surgery_level_field = None
            asa_grade_field = 'assessment_asa'
            age_field = 'patient_age'
        elif self.dataset_name == 'MOVER_SIS':
            surgery_level_field = None
            asa_grade_field = None
            age_field = 'patient_age'
        else:
            surgery_level_field = '手术等级'
            asa_grade_field = '总体评估时的ASA分级'
            age_field = '患者年龄'
        
        # 手术复杂度分析
        # surgery_level = extract_scalar(row.get(surgery_level_field, ''))
        # # MMSDP的level是数字，需要转换为中文
        # if self.dataset_name == 'Multimodal_SDP':
        #     level_map = {'1': '一类', '2': '二类', '3': '三类', '4': '四类'}
        #     surgery_level = level_map.get(str(surgery_level), str(surgery_level))
        
        # if surgery_level in self.medical_knowledge['surgery_complexity']:
        #     complexity_info = self.medical_knowledge['surgery_complexity'][surgery_level]
        #     context_parts.append(f"手术等级：{surgery_level}（{complexity_info['complexity']}手术，基准时长约{complexity_info['base_time']}分钟）")
        
        # ASA风险分析
        asa_grade = extract_scalar(row.get(asa_grade_field, ''))
        # MMSDP/INSPIRE/MOVER_EPIC 的 ASA 是数字，需要转换为中文；MOVER_SIS 无 ASA
        if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC'):
            asa_map = {'1': 'Ⅰ 级', '2': 'Ⅱ 级', '3': 'Ⅲ 级', '4': 'Ⅳ 级', '5': 'Ⅴ 级', '6': 'VI 级', '-1': '目前无评级'}
            asa_grade = asa_map.get(str(asa_grade), str(asa_grade))
        
        # if asa_grade in self.medical_knowledge['asa_risk_factors']:
        #     asa_info = self.medical_knowledge['asa_risk_factors'][asa_grade]
        #     context_parts.append(f"ASA分级：{asa_grade}（{asa_info['description']}，风险系数{asa_info['risk_multiplier']}）")
        
        # 年龄风险分析
        age_value = extract_scalar(row.get(age_field, ''))
        if age_value and age_value != '无':
            try:
                age = int(age_value)
                for age_group, info in self.medical_knowledge['age_risk_factors'].items():
                    if info['age_range'][0] <= age < info['age_range'][1]:
                        context_parts.append(f"年龄：{age}岁（{info['description']}，风险系数{info['risk_multiplier']}）")
                        break
            except (ValueError, TypeError):
                pass
        
        return "；".join(context_parts) if context_parts else ""
    
    def generate_doctor_context(self, row: pd.Series) -> str:
        try:
            context_parts = []
            surgeon_field = row.get('手术医生', '')
            
            if surgeon_field != '无' and not pd.isna(surgeon_field) and surgeon_field != 0 and str(surgeon_field) != '0':
                surgeon_ids = self._extract_doctor_ids(surgeon_field)
                for surgeon_id in surgeon_ids:
                    profile_key = f"surgeon_{surgeon_id}"
                    if profile_key in self.doctor_profiles:
                        profile = self.doctor_profiles[profile_key]
                        info = self._query_doctor_prior_stats(profile, row)
                        if info:
                            context_parts.append(f"手术医生(ID：{surgeon_id}){info}")
            
            return "\n".join(context_parts) if context_parts else ""
        except Exception as e:
            return ""
    
    def _query_doctor_prior_stats(self, profile: Dict, row: pd.Series) -> str:
        if not profile or 'priors' not in profile:
            return ""
        
        priors = profile['priors']
        
        v = lambda k: row.get(k, '')
        candidates = [
            (3, ['拟行手术名称','手术器械清点单手术名称','手术等级'], 
             (v('拟行手术名称'), v('手术器械清点单手术名称'), v('手术等级')),
             "拟行手术名称/手术器械清点单手术名称/手术等级"),
            (2, ['拟行手术名称','手术器械清点单手术名称'], 
             (v('拟行手术名称'), v('手术器械清点单手术名称')),
             "拟行手术名称/手术器械清点单手术名称"),
            (1, ['拟行手术名称'], 
             (v('拟行手术名称'),),
             "拟行手术名称"),
            (1, ['手术器械清点单手术名称'], 
             (v('手术器械清点单手术名称'),),
             "手术器械清点单手术名称"),
            (2, ['患者住院科室','手术等级'], 
             (v('患者住院科室'), v('手术等级')),
             "住院科室/手术等级")
        ]
    
        # 查找匹配的统计信息
        matched_info = ""
        for lvl, key_names, key_tuple, key_desc in candidates:
            key = (lvl,) + key_tuple
            # print(f"    尝试匹配: 键={key}, 描述={key_desc}")
            if key in priors:
                stats = priors[key]
                if stats.get('count', 0) >= 5:
                    # 构建具体的特征描述
                    feature_desc = self._build_feature_description(key_names, key_tuple)
                    
                    # 评估可靠性
                    reliability1, reliability2 = self._assess_doctor_reliability(lvl, stats, profile, key_desc)
                    
                    # 计算变异系数
                    cv = stats.get('std', 0) / stats.get('mean', 1) if stats.get('mean', 0) > 0 else 0
                    
                    # 根据变异系数评估能力稳定性
                    if cv <= 0.2:
                        stability_desc = "手术时长稳定性高"
                    elif cv <= 0.4:
                        stability_desc = "手术时长一般稳定"
                    elif cv <= 0.5:
                        stability_desc = "手术时长波动较大"
                    else:
                        stability_desc = "手术时长波动很大"
                    
                    return f"在相同（{feature_desc}）手术组合下，统计到的历史数据（共{stats['count']}例，{reliability1}）显示：\n• 中位时长：{stats['median']:.0f}分钟，平均时长：{stats['mean']:.0f}分钟\n• 时长范围：{stats['min']:.0f}-{stats['max']:.0f}分钟，四分位距：{stats['q25']:.0f}-{stats['q75']:.0f}分钟\n• {stability_desc}（变异系数{cv:.2f}），{reliability2}"

        return ""
    
    def _build_feature_description(self, key_names, key_tuple):
        """构建具体的特征描述"""
        descriptions = []
        for i, key_name in enumerate(key_names):
            if i < len(key_tuple):
                value = key_tuple[i]
                if key_name == '拟行手术名称':
                    descriptions.append(f"拟行手术名称={value}")
                elif key_name == '手术器械清点单手术名称':
                    descriptions.append(f"手术器械清点单手术名称={value}")
                elif key_name == '手术等级':
                    level_map = {'1': '一类', '2': '二类', '3': '三类', '4': '四类'}
                    level_desc = level_map.get(str(value), str(value))
                    descriptions.append(f"手术等级={level_desc}")
                elif key_name == '患者住院科室':
                    # 转换数字编号为科室名称
                    try:
                        dept_num = int(value)
                        if dept_num > 0 and dept_num <= len(hospital_departments):
                            dept_name = hospital_departments[dept_num - 1]
                            descriptions.append(f"患者住院科室={dept_name}")
                        else:
                            descriptions.append(f"患者住院科室=其他")
                    except (ValueError, TypeError):
                        descriptions.append(f"患者住院科室={value}")
                else:
                    descriptions.append(f"{key_name}={value}")
        
        return "，".join(descriptions)
    
    def _assess_doctor_reliability(self, match_level: int, stats: Dict, profile: Dict, key_desc: str) -> str:
        """评估医生先验信息的可靠性"""
        count = stats.get('count', 0)
        std = stats.get('std', 0)
        mean = stats.get('mean', 0)
        
        # 计算变异系数(CV)
        cv = std / mean if mean > 0 else 1.0
        
        # 获取医生总体经验
        total_cases = profile.get('total_cases', 0)
        overall_stats = profile.get('overall_stats', {})
        overall_cv = 0
        if overall_stats and overall_stats.get('mean', 0) > 0:
            overall_cv = overall_stats.get('std', 0) / overall_stats.get('mean', 0)
        
        # 可靠性评估规则
        if match_level >= 3 and count >= 8 and cv <= 0.3:
            return "可靠性较高", "强烈建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 3 and count >= 5 and cv <= 0.4:
            return "可靠性较高", "建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 2 and key_desc != "患者住院科室/手术等级" and count >= 10 and cv <= 0.4:
            return "可靠性较高",  "建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 2 and key_desc != "患者住院科室/手术等级" and count >= 5:
            return "可靠性高", "建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 2 and count >= 10 and key_desc == "患者住院科室/手术等级":
            return "可靠性中等", "建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 1 and count >= 10:
            return "可靠性高","建议参考该医生经验值并结合参考案例判断"
        elif match_level >= 1 and count >= 5:
            return "可靠性中等","建议参考该医生经验值并结合参考案例判断"
        else:
            return "可靠性较低","仅供参考"
    
    def _extract_doctor_ids(self, doctor_field):
        """从医生字段中提取所有医生ID"""
        try:
            # 适配 safe_get_value 的逻辑：0会被转换为"0"字符串
            if doctor_field == '无' or pd.isna(doctor_field) or doctor_field == 0 or str(doctor_field) == '0':
                return []
            
            if isinstance(doctor_field, (list, tuple)):
                return [str(x) for x in doctor_field if x != '无' and not pd.isna(x) and x != 0 and str(x) != '0']
            else:
                if doctor_field == 0 or str(doctor_field) == '0':
                    return []
                return [str(doctor_field)]
        except Exception as e:
            # 如果出现任何异常，返回空列表
            return []
    
    def calculate_reliability_score(self, stats: Dict) -> Tuple[str, float]:
        """计算先验信息的可靠性评分"""
        count = stats.get('count', 0)
        std_val = stats.get('std', 0)
        mean_val = stats.get('mean', 0)
        
        if mean_val <= 0:
            return "低", 0.0
        
        cv = std_val / mean_val  # 变异系数
        
        # 综合评分：样本数权重0.6，稳定性权重0.4
        sample_score = min(1.0, count / 30)  # 30个样本为满分
        stability_score = max(0.0, 1.0 - cv)  # 变异系数越小越好
        
        reliability_score = sample_score * 0.6 + stability_score * 0.4
        
        if reliability_score >= 0.8:
            return "高", reliability_score
        elif reliability_score >= 0.6:
            return "中", reliability_score
        else:
            return "低", reliability_score
    
    def generate_prior_hint(self, row: pd.Series, retrieved_cases: Optional[pd.DataFrame] = None) -> str:
        """生成完整的先验提示
        
        Args:
            row: 当前查询案例
            retrieved_cases: 检索到的相似案例（DataFrame），可选
        """
        found, stats, key_desc = self.query_prior_stats(row)
        
        if not found or not stats or stats.get('count', 0) == 0:
            # 即使没有历史统计信息，如果有检索案例，也可以添加检索案例统计
            if retrieved_cases is not None and len(retrieved_cases) > 0:
                retrieved_stats = self._calculate_retrieved_cases_stats(retrieved_cases)
                if retrieved_stats:
                    prior_hint = f"\n\n【检索案例统计提示】从相似案例检索中获得的统计信息：\n"
                    prior_hint += retrieved_stats
                    prior_hint += f"\n以上信息仅作保守参考，若参考案例显示明显差异，请以相似案例为准。"
                    return prior_hint
            return ""
        
        # 获取统计信息
        median_val = stats.get('median', 0)
        mean_val = stats.get('mean', 0)
        count_val = stats.get('count', 0)
        std_val = stats.get('std', 0)
        q25_val = stats.get('q25', 0)
        q75_val = stats.get('q75', 0)
        min_val = stats.get('min', 0)
        max_val = stats.get('max', 0)
        
        if median_val <= 0 or mean_val <= 0:
            # 即使历史统计无效，如果有检索案例，也可以添加检索案例统计
            if retrieved_cases is not None and len(retrieved_cases) > 0:
                retrieved_stats = self._calculate_retrieved_cases_stats(retrieved_cases)
                if retrieved_stats:
                    prior_hint = f"\n\n【检索案例统计提示】从相似案例检索中获得的统计信息：\n"
                    prior_hint += retrieved_stats
                    prior_hint += f"\n以上信息仅作保守参考，若参考案例显示明显差异，请以相似案例为准。"
                    return prior_hint
            return ""
        
        # 计算可靠性
        reliability, reliability_score = self.calculate_reliability_score(stats)
        
        # 生成医学上下文
        medical_context = self.generate_medical_context(row)
        
        # 构建先验提示
        prior_hint = f"\n\n【先验提示】在相同'{key_desc}'手术组合下，统计到的历史数据（共{count_val}例，可靠性{reliability}）显示：\n"
        prior_hint += f"• 中位时长：{median_val:.0f}分钟，平均时长：{mean_val:.0f}分钟\n"
        prior_hint += f"• 时长范围：{min_val:.0f}-{max_val:.0f}分钟，四分位距：{q25_val:.0f}-{q75_val:.0f}分钟\n"
        
        # 添加稳定性建议
        cv = std_val / mean_val if mean_val > 0 else 1.0
        if cv < 0.3:
            prior_hint += f"• 数据相对稳定（变异系数{cv:.2f}），建议参考中位数并结合参考案例判断\n"
        elif cv < 0.6:
            prior_hint += f"• 数据中等波动（变异系数{cv:.2f}），请结合参考案例判断\n"
        else:
            prior_hint += f"• 数据波动较大（变异系数{cv:.2f}），请谨慎参考\n"
        
        # 添加医学上下文
        if medical_context:
            prior_hint += f"• 患者医学特征：{medical_context}\n"
        
        # 添加检索案例统计信息（如果提供了检索案例）
        if retrieved_cases is not None and len(retrieved_cases) > 0:
            retrieved_stats = self._calculate_retrieved_cases_stats(retrieved_cases)
            if retrieved_stats:
                prior_hint += f"\n【检索案例统计】检索到相似度最高的Top-{len(retrieved_cases)}个中获得的统计信息：\n"
                prior_hint += retrieved_stats
        
        # # 添加医生团队上下文
        # doctor_context = self.generate_doctor_context(row)
        # if doctor_context:
        #     prior_hint += f"\n【医生经验提示】{doctor_context}\n"
        
        prior_hint += f"以上信息仅作保守参考，若参考案例显示明显差异，请以相似案例为准。"
        
        return prior_hint
    
    def _calculate_retrieved_cases_stats(self, retrieved_cases: pd.DataFrame) -> str:
        """计算检索案例的统计信息
        
        Args:
            retrieved_cases: 检索到的相似案例（DataFrame）
        
        Returns:
            str: 格式化的统计信息字符串，如果无法计算则返回空字符串
        """
        if retrieved_cases is None or len(retrieved_cases) == 0:
            return ""
        
        try:
            # 根据数据集类型选择时长字段
            if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                duration_field = 'duration'
            else:
                duration_field = '手术间时长'
            
            # 提取时长数据
            if duration_field not in retrieved_cases.columns:
                return ""
            
            durations = retrieved_cases[duration_field].values
            valid_durations = durations[~np.isnan(durations)]
            valid_durations = valid_durations[valid_durations > 0]
            
            if len(valid_durations) == 0:
                return ""
            
            # 计算统计信息
            mean_val = float(np.mean(valid_durations))
            median_val = float(np.median(valid_durations))
            std_val = float(np.std(valid_durations))
            min_val = float(np.min(valid_durations))
            max_val = float(np.max(valid_durations))
            q25_val = float(np.percentile(valid_durations, 25))
            q75_val = float(np.percentile(valid_durations, 75))
            count_val = len(valid_durations)
            
            # 构建统计信息字符串
            # stats_str = f"• 案例数量：{count_val}例\n"
            stats_str = f"• 平均时长：{mean_val:.0f}分钟，中位时长：{median_val:.0f}分钟\n"
            stats_str += f"• 时长范围：{min_val:.0f}-{max_val:.0f}分钟，四分位距：{q25_val:.0f}-{q75_val:.0f}分钟\n"
            
            # 计算变异系数
            cv = std_val / mean_val if mean_val > 0 else 1.0
            if cv < 0.3:
                stats_str += f"• 数据相对稳定（变异系数{cv:.2f}）\n"
            elif cv < 0.6:
                stats_str += f"• 数据中等波动（变异系数{cv:.2f}）\n"
            else:
                stats_str += f"• 数据波动较大（变异系数{cv:.2f}）\n"
            
            return stats_str
        except Exception as e:
            # 如果计算统计信息时出错，返回空字符串
            return ""
    
    def _contains_doctor(self, doctor_field, target_doctor_id):
        """检查医生字段是否包含目标医生ID"""
        # 安全处理列表类型的医生字段
        if isinstance(doctor_field, (list, tuple)):
            # 如果是列表，检查是否包含目标ID
            return str(target_doctor_id) in [str(x) for x in doctor_field if x != 0 and x != '无' and not pd.isna(x) and str(x) != '0']
        else:
            # 如果是单个值，检查是否有效且匹配
            if doctor_field == '无' or pd.isna(doctor_field) or doctor_field == 0 or str(doctor_field) == '0':
                return False
            return str(doctor_field) == str(target_doctor_id)
    
    def _safe_to_float(self, value, default=0.0):
        """安全转换为浮点数"""
        try:
            v = float(value)
            return v if v > 0 else default
        except Exception:
            return default