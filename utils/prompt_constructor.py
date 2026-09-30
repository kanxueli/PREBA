import pandas as pd
import os
import random
import numpy as np

# Our Dataset Dictory
Feature_Variable_Dict={
    "患者性别": "gender",
    "患者年龄": "age",
    "患者住院科室": "ward",
    "术前检查中肺功能是否异常": "lung_function_abnormal",
    "术前检查中血气是否异常": "blood_gas_abnormal",
    "术前检查中超声心动图是否异常": "echo_abnormal",
    "术前检查中放射结果详情描述": "radiology_details",
    "术前检查中心电图检查详细情况": "ecg_details",
    "术前检查中凝血筛选详细情况": "coagulation_details",
    "患者入手术室时间": "surgery_start_time",
    "患者详情中对应的手术类型": "surgery_type",
    "拟行手术名称": "planned_surgery_name",
    "手术器械清点单手术名称": "instrument_check_surgery_name",
    "麻醉前访视吸烟史标识": "smoking_history",
    "麻醉前访视酗酒史": "alcohol_history",
    "麻醉前访视呼吸系统病史": "resp_history",
    "麻醉前访视神经系统病史": "neuro_history",
    "麻醉前访视消化系统病史": "digest_history",
    "麻醉前访视脊柱四肢病史": "spine_limb_history",
    "麻醉前访视泌尿系统病史": "uro_history",
    "麻醉前访视免疫系统病史": "immune_history",
    "麻醉前访视其它系统病史异常情况": "other_sys_abn",
    "麻醉前访视-过敏史详情": "allergy_detail",
    "麻醉前访视查体-一般情况": "general_exam",
    "麻醉前访视查体-心肺听诊详情": "cardio_pulm_ausc",
    "麻醉前访视-手术史详情": "prior_surgery",
    "麻醉前访视病史治疗情况": "treatment_history",
    "手术排班的麻醉方法": "planned_anesthesia_method",
    "拟选用麻醉药": "planned_anesthesia_drug",
    "麻醉前访视麻醉史": "anesthesia_history",
    "麻醉计划风险评估时患者心功能分级(New York)": "nyha_class",
    "麻醉前访视中的特殊情况记录": "exam_notes",
    "手术医生": "surgeon",
    "麻醉医生姓名": "anesthesiologist",
    "手术排班的巡回护士": "circulating_nurse",
    "手术排班的洗手护士": "scrub_nurse",
    "手术排班的手术房间id": "surgery_room_id",
    "手术排班的排班台次": "scheduling_shift",
    # "手术排班完成台次": "completed_shift",
    "手术等级": "surgery_level",
    "主要诊断": "primary_diagnosis",
    "手术注意事项": "surgery_notes",
    "总体评估时的ASA分级": "asa_grade",
    "手术间时长": "ground_true"
    }

hospital_departments = [
    "骨科",
    "妇科",
    "产科",
    "乳腺甲状腺外科",
    "生殖妇科",
    "普外一科",
    "泌尿外科",
    "普外二科",
    "胸外科",
    "肝胆胰外科",
    "耳鼻咽喉科",
    "急诊创伤外科",
    "口腔颌面外科",
    "心脏大血管外科",
    "神经内科",
    "肾内科",
    "重症医学科",
    "肛肠科",
    "日间病房中心",
    "眼科",
    "疼痛科",
    "内科重症监护病房", # MICU
    "消化内科",
    "康复医学科",
    "肿瘤内科",
    "老年医学科干部保健科四病区",
    "儿科",
    "呼吸与危重症医学科",
    "血液内科",
    "急诊内科",
    "心血管内科",
    "感染疾病及肝病科",
    "老年医学科干部保健科二病区",
    "老年医学科干部保健科一病区",
    "全科医学科",
    "老年医学科干部保健科综合一病区",
    "风湿免疫科",
    "脑卒中中心",
    "生殖医学科",
    "临床心理科",
]


# 系统Prompt定义角色
# System_prompt = "你是一个手术时长预测机器人，旨在根据用户提供的患者生理特征信息、手术类型、医疗团队和相关参考实例等，来预测该台手术的时间长度。"
System_prompt = "你是一个手术时长预测机器人，旨在根据医学常识和用户提供的患者生理特征、手术类型和相关参考实例等，来预测该台手术的时间长度。以下是你需要关注的五个主要类别特征: \
\n  1. 患者基本信息与健康状况: 包含患者的基本生理特征、住院科室以及术前的健康检查信息等。\
\n  2. 手术类型和手术本身特征: 包括手术的名称、类型、等级和主要诊断等。\
\n  3. 麻醉与术前访视评估: 包括麻醉前访视时对患者的详细病史评估，包括呼吸、神经、消化系统等的病史等。\
\n  4. 医疗团队与支持人员: 包括手术的医生、护士以及其他支持人员的安排。\
\n  5. 手术紧急程度及特殊情况: 包括急诊手术标识、手术注意事项等特殊情况。\
\n请注意，你的输出必须是中文且按照格式输出。"

# Zero_shot_System_prompt
# System_prompt = "你是一个手术时长预测机器人，旨在根据医学常识和用户提供的患者生理特征、手术类型等，来预测该台手术的时间长度。以下是你需要关注的五个主要类别特征: \
# \n  1. 患者基本信息与健康状况: 包含患者的基本生理特征、住院科室以及术前的健康检查信息等。\
# \n  2. 手术类型和手术本身特征: 包括手术的名称、类型、等级和主要诊断等。\
# \n  3. 麻醉与术前访视评估: 包括麻醉前访视时对患者的详细病史评估，包括呼吸、神经、消化系统等的病史等。\
# \n  4. 医疗团队与支持人员: 包括手术的医生、护士以及其他支持人员的安排。\
# \n  5. 手术紧急程度及特殊情况: 包括急诊手术标识、手术注意事项等特殊情况。\
# \n请注意，你只能输出手术时长数值(例如：100)，不要输出其他任何内容。"
# # \n请注意，你只需要输出手术时长数值，不需要输出其他任何内容。"

# Prompt Design For MMSDP Dataset
System_prompt_MMSDP = "你是一个手术时长预测机器人，旨在根据医学常识和用户提供的患者生理特征、手术类型和相关参考实例等，来预测该台手术的时间长度。以下是你需要关注的三类主要特征: \
\n  1. 患者基本信息与健康检查情况: 包含患者的基本生理特征、住院科室以及术前的健康检查信息等。\
\n  2. 手术基本信息: 包括手术的名称、类型、等级和主要诊断等。\
\n  3. 术前访视评估与麻醉相关特征: 包括麻醉前访视时对患者的详细病史评估，包括呼吸、神经、消化系统等的病史等。\
\n请注意，你的输出必须是中文且按照格式输出。"

# Zero_shot_System_prompt_MMSDP
# System_prompt_MMSDP = "你是一个手术时长预测机器人，旨在根据医学常识和用户提供的患者生理特征、手术类型等，来预测该台手术的时间长度。以下是你需要关注的三类主要特征: \
# \n  1. 患者基本信息与健康检查情况: 包含患者的基本生理特征、住院科室以及术前的健康检查信息等。\
# \n  2. 手术基本信息: 包括手术的名称、类型、等级和主要诊断等。\
# \n  3. 术前访视评估与麻醉相关特征: 包括麻醉前访视时对患者的详细病史评估，包括呼吸、神经、消化系统等的病史等。\
# \n请注意格式输出指令：你只需要输出手术时长数值，例如100，不要输出其他任何内容，否则视为无效输出。" #for deepseek
# # \n请注意，你只需要输出手术时长数值，不需要输出其他任何内容。" # for huatuogpt

# Prompt Design For INSPIRE Dataset (English; aligns with example_template_INSPIRE.txt)
System_prompt_INSPIRE = "You are a surgery duration prediction assistant. Given patient and surgery information plus several similar reference cases, predict the duration of this surgery in minutes. \
\nInput has two parts; use them together with reference case durations to give a reasonable prediction: \
\n  1. Patient basic information: gender, age, height/weight, department. \
\n  2. Diagnosis and surgery: ASA grade, emergency surgery (yes/no), primary diagnosis text, ICD code, preop labs. \
\nOutput must follow the required format strictly (e.g. output only the numeric duration or as specified)."

# Zero_shot_System_prompt for INSPIRE Dataset:
# System_prompt_INSPIRE = "You are a surgery duration prediction assistant. Given patient and surgery information plus several similar reference cases, predict the duration of this surgery in minutes. \
# \nInput has two parts; use them together with reference case durations to give a reasonable prediction: \
# \n  1. Patient basic information: gender, age, height/weight, department. \
# \n  2. Diagnosis and surgery: ASA grade, emergency surgery (yes/no), primary diagnosis text, ICD code, preop labs. \
# \nPlease note that you can only output the numeric duration (e.g. 100), do not output any other text or symbols."

# MOVER_EPIC / MOVER_SIS: English surgery duration prediction (same style as INSPIRE)
System_prompt_MOVER_EPIC = "You are a surgery duration prediction assistant. Given patient and procedure information plus several similar reference cases, predict the duration of this surgery in minutes. \
\nUse patient demographics, procedure name, diagnosis names and ASA grade together with reference case durations to give a reasonable prediction. \
\nOutput must follow the required format strictly (e.g. output only the numeric duration or as specified)."

# Zero_shot_System_prompt_MOVER_EPIC / MOVER_SIS:
# System_prompt_MOVER_EPIC = "You are a surgery duration prediction assistant. Given patient and procedure information plus several similar reference cases, predict the duration of this surgery in minutes. \
# \nUse patient demographics, procedure name, diagnosis names and ASA grade together with reference case durations to give a reasonable prediction. \
# \nPlease note that you can only output the numeric duration (e.g. 100), do not output any other text or symbols. "

System_prompt_MOVER_SIS = "You are a surgery duration prediction assistant. Given patient and procedure information plus several similar reference cases, predict the duration of this surgery in minutes. \
\nUse patient demographics, procedure name and preop medications together with reference case durations to give a reasonable prediction. \
\nOutput must follow the required format strictly (e.g. output only the numeric duration or as specified)."


def safe_get_value(row, key, default=''):
    """安全获取数据值，处理空值和数组类型"""
    value = row.get(key, default)
    
    # 处理None值
    if value is None:
        return '无'
    
    # 处理数组 / 序列类型的值（可能嵌套 list/ndarray/Series），统一展开为标量再判断
    if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
        if len(value) == 0:
            return '无'
        # 如果序列中有非空值，取第一个非空标量值
        for item in value:
            v = item
            # 展平可能嵌套的 list / tuple / ndarray / Series，直到拿到标量
            while isinstance(v, (list, tuple, pd.Series, np.ndarray)):
                if len(v) == 0:
                    v = None
                    break
                # 对 Series 使用 iloc[0]，对其他序列使用下标 0
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                else:
                    v = v[0]
            if v is None:
                continue
            # 此时 v 应为标量，可以安全使用 pd.isna
            if not pd.isna(v) and str(v).strip() not in ['', 'nan']:
                return str(v)
        return '无'
    
    # 处理单个值
    if pd.isna(value) or value == -1 or value == -2 or value == "-2":
        return '无'
    
    # 转换为字符串并检查
    str_value = str(value).strip()
    if str_value == '' or str_value == 'nan' or str_value == 'None':
        return '无'
    
    return str_value

def create_variables_dict(row, example_index=None, include_ground_true=True):
    """创建变量字典，供不同prompt构造函数使用"""
    variables = {
        'gender': {'0': '男', '1': '女'}.get(safe_get_value(row, '患者性别'), '无'),
        'age': safe_get_value(row, '患者年龄'),
        'ward': hospital_departments[int(safe_get_value(row, '患者住院科室'))-1] if int(safe_get_value(row, '患者住院科室')) > 0 else '其他',
        'lung_function_abnormal': "否" if safe_get_value(row, '术前检查中肺功能是否异常') == "无" else safe_get_value(row, '术前检查中肺功能是否异常'),
        'blood_gas_abnormal': "否" if safe_get_value(row, '术前检查中血气是否异常') == "无" else safe_get_value(row, '术前检查中血气是否异常'),
        'echo_abnormal': "否" if safe_get_value(row, '术前检查中超声心动图是否异常') == "无" else safe_get_value(row, '术前检查中超声心动图是否异常'),
        'radiology_details': safe_get_value(row, '术前检查中放射结果详情描述'),
        'ecg_details': safe_get_value(row, '术前检查中心电图检查详细情况'),
        'coagulation_details': safe_get_value(row, '术前检查中凝血筛选详细情况'),
        'surgery_start_time': safe_get_value(row, '患者入手术室时间'),
        'surgery_type': {'1': '择期手术', '2': '急诊手术', '0': '其他'}.get(safe_get_value(row, '患者详情中对应的手术类型'), '无'),
        'planned_surgery_name': safe_get_value(row, '拟行手术名称'),
        'instrument_check_surgery_name': safe_get_value(row, '手术器械清点单手术名称'),
        'smoking_history': {"True": "是", "False": "否"}.get(safe_get_value(row, '麻醉前访视吸烟史标识'), '否'),
        'alcohol_history': {"True": "是", "False": "否"}.get(safe_get_value(row, '麻醉前访视酗酒史'), '否'),
        'resp_history': safe_get_value(row, '麻醉前访视呼吸系统病史'),
        'neuro_history': safe_get_value(row, '麻醉前访视神经系统病史'),
        'digest_history': safe_get_value(row, '麻醉前访视消化系统病史'),
        'spine_limb_history': safe_get_value(row, '麻醉前访视脊柱四肢病史'),
        'uro_history': safe_get_value(row, '麻醉前访视泌尿系统病史'),
        'immune_history': safe_get_value(row, '麻醉前访视免疫系统病史'),
        'other_sys_abn': safe_get_value(row, '麻醉前访视其它系统病史异常情况'),
        'allergy_detail': safe_get_value(row, '麻醉前访视-过敏史详情'),
        'general_exam': safe_get_value(row, '麻醉前访视查体-一般情况'),
        'cardio_pulm_ausc': safe_get_value(row, '麻醉前访视查体-心肺听诊详情'),
        'prior_surgery': safe_get_value(row, '麻醉前访视-手术史详情'),
        'treatment_history': safe_get_value(row, '麻醉前访视病史治疗情况'),
        'planned_anesthesia_method': safe_get_value(row, '手术排班的麻醉方法'),
        'planned_anesthesia_drug': safe_get_value(row, '拟选用麻醉药'),
        'anesthesia_history': "无" if int(safe_get_value(row, '麻醉前访视麻醉史')) < 0 else "有",
        'nyha_class': {1: 'Ⅰ 级', '2': 'Ⅱ 级', '3': 'Ⅲ 级', '4': 'Ⅳ 级'}.get(safe_get_value(row, '麻醉计划风险评估时患者心功能分级(New York)'), '目前无评级'),
        'exam_notes': safe_get_value(row, '麻醉前访视中的特殊情况记录'),
        'surgeon': safe_get_value(row, '手术医生'),
        'anesthesiologist': safe_get_value(row, '麻醉医生姓名'),
        'circulating_nurse': safe_get_value(row, '手术排班的巡回护士'),
        'scrub_nurse': safe_get_value(row, '手术排班的洗手护士'),
        'surgery_room_id': safe_get_value(row, '手术排班的手术房间id'),
        'scheduling_shift': safe_get_value(row, '手术排班的排班台次'),
        'surgery_level': {'1': '一类', '2': '二类', '3': '三类', '4': '四类'}.get(safe_get_value(row, '手术等级'), '目前无评级'),
        'primary_diagnosis': safe_get_value(row, '主要诊断'),
        'surgery_notes': safe_get_value(row, '手术注意事项'),
        'asa_grade': {'-1': '目前无评级', '1': 'Ⅰ 级', '2': 'Ⅱ 级', '3': 'Ⅲ 级', '4': 'Ⅳ 级', '5': 'Ⅴ 级', '6': 'VI 级'}.get(safe_get_value(row, '总体评估时的ASA分级'), '无')
    }
    
    # 添加example_index（如果提供）
    if example_index is not None:
        variables['example_index'] = example_index
    
    # 添加ground_true（如果需要）
    if include_ground_true:
        variables['ground_true'] = safe_get_value(row, '手术间时长')
    
    return variables

def create_variables_dict_MMSDP(row, example_index=None, include_ground_true=True):
    """
    为MMSDP数据集创建变量字典
    
    Args:
        row: 单行数据（pandas Series）
        example_index: 示例索引（可选）
        include_ground_true: 是否包含真实手术时长
    
    Returns:
        dict: 变量字典
    """
    # 注意：模板中第9行使用了diagnostic_CBC_RBC两次，第二次应该是血红蛋白
    # 但为了与模板保持一致，我们按照模板中的字段名来创建
    # 2026新MMSDP数据集特征：diagnostic_electrolyte, diagnoses, procedures, assessment_asa, surgeons, anesthesiologists
    variables = {
        'patient_gender': safe_get_value(row, 'patient_gender', '无'),
        'patient_age': safe_get_value(row, 'patient_age', '无'),
        'patient_height': safe_get_value(row, 'patient_height', '无'),
        'patient_weight': safe_get_value(row, 'patient_weight', '无'),
        'department': safe_get_value(row, 'department', '无'),
        'diagnostic_electrolyte': {"abnormal": "是", "normal": "否"}.get(safe_get_value(row, 'diagnostic_electrolyte', '否'), '否'),
        'diagnostic_ECG': {"abnormal": "是", "normal": "否"}.get(safe_get_value(row, 'diagnostic_ECG', '否'), '否'),
        'diagnostic_liver': {"abnormal": "是", "normal": "否"}.get(safe_get_value(row, 'diagnostic_liver', '否'), '否'),
        'diagnostic_kidney': {"abnormal": "是", "normal": "否"}.get(safe_get_value(row, 'diagnostic_kidney', '否'), '否'),
        'diagnostic_CBC_RBC': safe_get_value(row, 'diagnostic_CBC_RBC', '无'),
        'diagnostic_CBC_WBC': safe_get_value(row, 'diagnostic_CBC_WBC', '无'),
        'diagnostic_CBC_HGB': safe_get_value(row, 'diagnostic_CBC_HGB', '无'),
        'diagnostic_CBC_PLT': safe_get_value(row, 'diagnostic_CBC_PLT', '无'),
        'diagnoses': safe_get_value(row, 'diagnoses', '无'),
        'procedures': safe_get_value(row, 'procedures', '无'),
        'level': safe_get_value(row, 'level', '无'),
        'assessment_asa': {'-1': '目前无评级', '1': 'Ⅰ 级', '2': 'Ⅱ 级', '3': 'Ⅲ 级', '4': 'Ⅳ 级', '5': 'Ⅴ 级', '6': 'VI 级','Ⅰ': 'Ⅰ 级', 'Ⅱ': 'Ⅱ 级', 'Ⅲ': 'Ⅲ 级', 'Ⅳ': 'Ⅳ 级', }.get(safe_get_value(row, 'assessment_asa', '目前无评级'), '目前无评级'),
        'type': {'SELECTIVE_SURGERY': '择期手术', 'EMERGENCY_SURGERY': '急诊手术', 'OTHER': '其他'}.get(safe_get_value(row, 'type', '其他'), '其他'),
        'position': safe_get_value(row, 'position', '无'),
        'ICD': safe_get_value(row, 'ICD', '无'),
        'surgeons': safe_get_value(row, 'surgeons', '无'),
        'anesthesiologists': safe_get_value(row, 'anesthesiologists', '无'),
        'anesthesia': safe_get_value(row, 'anesthesia', '无'),
        'disease_allergies': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_allergies', 'false'), '否'),
        'degree_of_mouth_opening': {"good": "良好", "1": "差", "2": "差", "3": "中度", "4": "良好", "middle": "中度", "poor": "差"}.get(safe_get_value(row, 'degree_of_mouth_opening', 'good'), '良好'),
        'neck_activity': {"good": "良好", "normal": "良好", "middle": "中度", "poor": "差"}.get(safe_get_value(row, 'neck_activity', 'good'), '良好'),
        'disease_smoking': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_smoking', 'false'), '否'),
        'disease_alcoholism': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_alcoholism', 'false'), '否'),
        'disease_respiratory': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_respiratory', 'false'), '否'),
        'disease_nerve': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_nerve', 'false'), '否'),
        'disease_circulatory': {"FALSE": "否", "TRUE": "是"}.get(safe_get_value(row, 'disease_circulatory', 'false'), '否'),
    }
    
    # 添加example_index（如果提供）
    if example_index is not None:
        variables['example_index'] = example_index
    
    # 添加duration（如果需要）
    if include_ground_true:
        variables['duration'] = safe_get_value(row, 'duration', '无')
    
    return variables


def create_variables_dict_INSPIRE(row, example_index=None, include_ground_true=True):
    """
    Build variable dict for INSPIRE
    """
    asa_val = safe_get_value(row, 'assessment_asa', '')
    asa_map = {'-1': 'No grade', '1': 'Grade I', '2': 'Grade II', '3': 'Grade III', '4': 'Grade IV', '5': 'Grade V', '6': 'Grade VI'}
    assessment_asa = asa_map.get(str(asa_val), 'No grade' if asa_val in (None, '', -1) else str(asa_val))

    emop_val = safe_get_value(row, 'emop', 0)
    if emop_val in (1, '1', True, 'true', 'True', 'TRUE'):
        emop = 'Yes'
    else:
        emop = 'No'

    preop_raw = safe_get_value(row, 'preop_labs_latest_json', '')
    if preop_raw and str(preop_raw).strip() not in ('', '{}', 'nan'):
        s = str(preop_raw)[:200]
        if len(str(preop_raw)) > 200:
            s += '...'
        preop_labs_summary = s
    else:
        preop_labs_summary = 'None'

    default_na = 'N/A'
    variables = {
        'patient_gender': safe_get_value(row, 'patient_gender', default_na),
        'patient_age': safe_get_value(row, 'patient_age', default_na),
        'patient_height': safe_get_value(row, 'patient_height', default_na),
        'patient_weight': safe_get_value(row, 'patient_weight', default_na),
        'department': safe_get_value(row, 'department.1', default_na),
        'assessment_asa': assessment_asa,
        'emop': emop,
        'diagnoses_text': safe_get_value(row, 'diagnoses_text', default_na),
        'ICD': safe_get_value(row, 'ICD', default_na),
        'preop_labs_summary': preop_labs_summary,
    }
    if example_index is not None:
        variables['example_index'] = example_index
    if include_ground_true:
        variables['duration'] = safe_get_value(row, 'duration', default_na)
    # Keep English-only: safe_get_value returns '无' for None; normalize to N/A for INSPIRE
    for k in variables:
        if variables[k] == '无':
            variables[k] = 'N/A'
    return variables


def create_variables_dict_MOVER_EPIC(row, example_index=None, include_ground_true=True):
    """Build variable dict for MOVER_EPIC (procedure_name, diagnosis_names, assessment_asa; no patient_weight)."""
    default_na = 'N/A'
    asa_val = safe_get_value(row, 'assessment_asa', '')
    asa_map = {'-1': 'No grade', '1': 'Grade I', '2': 'Grade II', '3': 'Grade III', '4': 'Grade IV', '5': 'Grade V', '6': 'Grade VI'}
    assessment_asa = asa_map.get(str(asa_val), 'No grade' if asa_val in (None, '', -1) else str(asa_val))
    variables = {
        'patient_gender': safe_get_value(row, 'patient_gender', default_na),
        'patient_age': safe_get_value(row, 'patient_age', default_na),
        'patient_height': safe_get_value(row, 'patient_height', default_na),
        'assessment_asa': assessment_asa,
        'procedure_name': safe_get_value(row, 'procedure_name', default_na),
        'diagnosis_names': safe_get_value(row, 'diagnosis_names', default_na),
        'surgery_start_time': safe_get_value(row, 'surgery_start_time', default_na),
    }
    if example_index is not None:
        variables['example_index'] = example_index
    if include_ground_true:
        variables['duration'] = safe_get_value(row, 'duration', default_na)
    for k in variables:
        if variables[k] == '无':
            variables[k] = 'N/A'
    return variables


def create_variables_dict_MOVER_SIS(row, example_index=None, include_ground_true=True):
    """Build variable dict for MOVER_SIS (procedure_name, preop_medications; no ordinal)."""
    default_na = 'N/A'
    variables = {
        'patient_gender': safe_get_value(row, 'patient_gender', default_na),
        'patient_age': safe_get_value(row, 'patient_age', default_na),
        'patient_height': safe_get_value(row, 'patient_height', default_na),
        'patient_weight': safe_get_value(row, 'patient_weight', default_na),
        'procedure_name': safe_get_value(row, 'procedure_name', default_na),
        'preop_medications': safe_get_value(row, 'preop_medications', default_na),
        'surgery_start_time': safe_get_value(row, 'surgery_start_time', default_na),
    }
    if example_index is not None:
        variables['example_index'] = example_index
    if include_ground_true:
        variables['duration'] = safe_get_value(row, 'duration', default_na)
    for k in variables:
        if variables[k] == '无':
            variables[k] = 'N/A'
    return variables


def fixed_reference_instance_construction(df, example_number=10, random_seed=42, dataset_name=''):
    """
    从数据框中随机选择example_number条数据来构建参考示例
    
    Args:
        df: 训练数据框
        example_number: 参考示例数量，默认为10
        random_seed: 随机种子，默认为42，确保可复现
    
    Returns:
        str: 构建好的参考示例字符串
    """
    # 设置随机种子确保可复现
    random.seed(random_seed)
    
    # 随机选择指定数量的数据
    if len(df) < example_number:
        example_number = len(df)
    
    selected_indices = random.sample(range(len(df)), example_number)
    selected_data = df.iloc[selected_indices]
    
    # 读取模板文件
    if dataset_name == 'Multimodal_SDP':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MMSDP.txt')
    elif dataset_name == 'INSPIRE':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_INSPIRE.txt')
    elif dataset_name == 'MOVER_EPIC':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_EPIC.txt')
    elif dataset_name == 'MOVER_SIS':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_SIS.txt')
    else:
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # 构建参考示例
    examples = []
    for i, (idx, row) in enumerate(selected_data.iterrows()):
        if dataset_name == 'Multimodal_SDP':
            variables = create_variables_dict_MMSDP(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'INSPIRE':
            variables = create_variables_dict_INSPIRE(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'MOVER_EPIC':
            variables = create_variables_dict_MOVER_EPIC(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'MOVER_SIS':
            variables = create_variables_dict_MOVER_SIS(row, example_index=i+1, include_ground_true=True)
        else:
            variables = create_variables_dict(row, example_index=i+1, include_ground_true=True)
        example = template.format(**variables)
        examples.append(example)
    
    return '\n\n'.join(examples)

def rag_reference_instance_construction(retrieved_cases, similarity_scores=None, dataset_name=''):
    """
    从RAG检索结果构建参考示例，按相似度排序
    
    Args:
        retrieved_cases: RAG检索到的案例数据框
        similarity_scores: 相似度分数列表（可选）
        dataset_name: 数据集名称，用于选择正确的模板和变量创建函数
    
    Returns:
        str: 构建好的参考示例字符串
    """
    if len(retrieved_cases) == 0:
        return ""
    
    # 读取模板文件
    if dataset_name == 'Multimodal_SDP':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MMSDP.txt')
    elif dataset_name == 'INSPIRE':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_INSPIRE.txt')
    elif dataset_name == 'MOVER_EPIC':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_EPIC.txt')
    elif dataset_name == 'MOVER_SIS':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_SIS.txt')
    else:
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # 构建参考示例
    examples = []
    for i, (idx, row) in enumerate(retrieved_cases.iterrows()):
        if dataset_name == 'Multimodal_SDP':
            variables = create_variables_dict_MMSDP(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'INSPIRE':
            variables = create_variables_dict_INSPIRE(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'MOVER_EPIC':
            variables = create_variables_dict_MOVER_EPIC(row, example_index=i+1, include_ground_true=True)
        elif dataset_name == 'MOVER_SIS':
            variables = create_variables_dict_MOVER_SIS(row, example_index=i+1, include_ground_true=True)
        else:
            variables = create_variables_dict(row, example_index=i+1, include_ground_true=True)
        
        # 如果有相似度分数，添加到变量中
        if similarity_scores and i < len(similarity_scores):
            variables['similarity'] = f" (相似度: {similarity_scores[i]:.3f})"
        else:
            variables['similarity'] = ""
        
        # 使用模板生成示例
        example = template.format(**variables)
        examples.append(example)
        
        # # 打印RAG案例信息
        # query_department = safe_get_value(row, '患者住院科室', '未知科室')
        # surgery_name = safe_get_value(row, '拟行手术名称', '未知手术')
        # duration = safe_get_value(row, '手术间时长', '未知时长')
        # surgery_level = safe_get_value(row, '手术等级', '目前无评级')
        # primary_diagnosis = safe_get_value(row, '主要诊断', '无')
        # instrument_check_surgery_name = safe_get_value(row, '手术器械清点单手术名称', '无')
        # asa_grade = safe_get_value(row, '总体评估时的ASA分级', '目前无评级')
        # nyha_class = safe_get_value(row, '麻醉计划风险评估时患者心功能分级(New York)', '目前无评级')
        # print(f"  相似案例 {i+1}: 科室={query_department}, 时长={duration}min, 主要诊断={primary_diagnosis}, 拟行手术名称={surgery_name}, 手术器械清点单手术名称={instrument_check_surgery_name}, 手术等级={surgery_level}, ASA分级={asa_grade}, 患者心功能分级={nyha_class}{variables['similarity']}")
    
    # 将所有示例连接起来
    return '\n\n'.join(examples)

def to_be_predicted_instance_construction(row, dataset_name=''):
    """
    构建待预测实例
    
    Args:
        row: 单行数据（pandas Series）
        dataset_name: 数据集名称，用于选择正确的模板和变量创建函数
    
    Returns:
        tuple: (待预测实例的prompt字符串, 真实手术时长值float)
    """
    # 读取模板文件
    if dataset_name == 'Multimodal_SDP':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MMSDP.txt')
    elif dataset_name == 'INSPIRE':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_INSPIRE.txt')
    elif dataset_name == 'MOVER_EPIC':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_EPIC.txt')
    elif dataset_name == 'MOVER_SIS':
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template_MOVER_SIS.txt')
    else:
        template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'example_template.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    if dataset_name == 'Multimodal_SDP':
        variables = create_variables_dict_MMSDP(row, example_index=None, include_ground_true=False)
        ground_true_value = safe_get_value(row, 'duration', '无')
    elif dataset_name == 'INSPIRE':
        variables = create_variables_dict_INSPIRE(row, example_index=None, include_ground_true=False)
        ground_true_value = safe_get_value(row, 'duration', '无')
    elif dataset_name == 'MOVER_EPIC':
        variables = create_variables_dict_MOVER_EPIC(row, example_index=None, include_ground_true=False)
        ground_true_value = safe_get_value(row, 'duration', '无')
    elif dataset_name == 'MOVER_SIS':
        variables = create_variables_dict_MOVER_SIS(row, example_index=None, include_ground_true=False)
        ground_true_value = safe_get_value(row, 'duration', '无')
    else:
        variables = create_variables_dict(row, example_index=None, include_ground_true=False)
        ground_true_value = safe_get_value(row, '手术间时长', '无')
    
    try:
        ground_true_float = float(ground_true_value) if ground_true_value != '无' else 0.0
    except (ValueError, TypeError):
        ground_true_float = 0.0
    
    # 使用模板生成待预测实例，去掉示例索引与真实时长
    if dataset_name == 'Multimodal_SDP':
        modified_template = template.replace('示例 {example_index}：\n', '').replace('手术时长：{duration}', '手术时长：')
    elif dataset_name in ('INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
        modified_template = template.replace('Example {example_index}:\n', '').replace('Surgery duration: {duration}', 'Surgery duration:')
    else:
        modified_template = template.replace('示例 {example_index}：\n', '').replace('手术时长：{ground_true}', '手术时长：')
    
    prompt = modified_template.format(**variables)
    
    return prompt, ground_true_float

def full_prompt_construction(demon_example, question, prior_hint, dataset_name=''):
    """
    构建完整的prompt，包含参考示例和待预测实例
    
    Args:
        demon_example (str): 参考示例字符串
        question (str): 待预测实例字符串
        prior_hint (str): 先验提示字符串，默认为空
    
    Returns:
        str: 完整的prompt字符串
    """
    if dataset_name in ('INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
        if demon_example == "":
            full_prompt = (
                prior_hint +
                "Below is the case to predict. Please predict the surgery duration (in minutes) based on the case and medical knowledge. Output only the numeric value (e.g. 100), no other content.\n" +
                question
            )
        else:
            full_prompt = (
                "Below are several similar historical cases with their actual surgery durations. Analyze their features and how they compare to the case to predict.\n" +
                demon_example +
                prior_hint +
                "\n\nBased on the above reference cases, reason by analogy and predict the duration for the current case (in minutes). Consider generality and specificity of the references, then output only the predicted number, no other content.\n" +
                question
            )
    else:
        if demon_example == "":
            full_prompt = (
                prior_hint +
                "以下是待预测实例，请根据患者的实际情况和医学常识来预测手术时长(请严格遵守格式输出指令：仅需要输出数值（例如：100），不要输出其他任何内容，否则视为无效输出)：\n" +
                question
            )
        else:
            full_prompt = (
                "以下是若干台与当前手术相似的历史手术案例及其真实手术时长。请仔细分析这些参考案例的特征，并思考它们与待预测案例的相似与不同之处。\n" +
                demon_example +
                prior_hint +
                "\n\n现在请基于以上参考案例的分析，类比推理当前手术的预计时长，预测时需要考虑参考案例的一般性和特异性，并结合待预测案例的实际情况进行综合考虑，然后保守预测，不宜过于激进。请仅输出最终的预测数值（单位：分钟），不需输出任何其他内容。\n" +
                question
            )
    return full_prompt