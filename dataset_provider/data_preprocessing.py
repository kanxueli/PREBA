
import pandas as pd 
import numpy as np
import re
from datetime import datetime
import pickle
import json


class SurgeryDataPreprocessor:
    def __init__(self):
        """初始化预处理器"""
        # 科室编码字典 - 从Duration.py中获取
        self.departments = [
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
            "MICU",
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

        # 医生编码映射
        self.surgeon_encoding = {}
        self.anesthesiologist_encoding = {}

    def load_data(self, file_path):
        """加载pkl文件"""
        print(f"正在加载数据文件: {file_path}")
        df = pd.read_pickle(file_path)  # 改为读取pkl文件
        print(f"数据加载完成，共{len(df)}行，{len(df.columns)}列")
        return df

    def preprocess_column_1_unique_id(self, series):
        """
        处理第1列：手麻系统中手术状态时间历史记录唯一标识
        直接返回，作为记录的唯一标识符
        """
        return series

    def preprocess_column_2_room_entry_time(self, series):
        """
        处理第2列：患者入手术室时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_3_prep_time(self, series):
        """
        处理第3列：术前准备时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_4_surgery_room_entry(self, series):
        """
        处理第4列：患者入手术间时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_5_anesthesia_start(self, series):
        """
        处理第5列：麻醉开始时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_6_surgery_start(self, series):
        """
        处理第6列：手术开始时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_7_surgery_end(self, series):
        """
        处理第7列：手术结束时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_8_anesthesia_end(self, series):
        """
        处理第8列：麻醉结束时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_9_room_exit_time(self, series):
        """
        处理第9列：患者出手术间时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_10_completed_cases(self, series):
        """
        处理第10列：手术排班完成台次
        转换为数值类型，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_11_prediction_time(self, series):
        """
        处理第11列：术后时长预测时间
        转换为数值类型，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_12_unplanned_second_surgery(self, series):
        """
        处理第12列：非计划二次手术标识
        转换为布尔值
        """

        def convert_bool(x):
            if pd.isna(x):
                return False
            if isinstance(x, bool):
                return x
            x_str = str(x).strip().lower()
            return x_str in ["true", "1", "yes", "是", "t"]

        return series.apply(convert_bool)

    def preprocess_column_13_patient_id(self, series):
        """
        处理第13列：院内临床信息患者唯一标识
        直接返回作为患者标识符
        """
        return series

    def preprocess_column_14_patient_gender(self, series):
        """
        处理第14列：患者性别
        编码为数值：女=1, 男=0, 缺失=-1
        """

        def encode_gender(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if x_str == "女":
                return 1
            elif x_str == "男":
                return 0
            else:
                return -1

        return series.apply(encode_gender)

    def preprocess_column_15_patient_age(self, series):
        """
        处理第15列：患者年龄
        提取年龄数值，支持"岁"和"个月"格式
        """

        def extract_age(x):
            if pd.isna(x):
                return -1

            x_str = str(x).strip()

            # 尝试直接转换数值
            try:
                return float(x_str)
            except:
                pass

            # 使用正则表达式提取年龄
            pattern = r"^(?:(\d+(?:\.\d+)?)\s*岁)?(?:(\d+(?:\.\d+)?)\s*个?月)?$"
            match = re.search(pattern, x_str)

            if match:
                years = float(match.group(1)) if match.group(1) else 0
                months = float(match.group(2)) if match.group(2) else 0
                return round(years + months / 12.0, 1)

            return 0

        return series.apply(extract_age)

    def preprocess_column_16_patient_height(self, series):
        """
        处理第16列：患者身高
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_17_patient_weight(self, series):
        """
        处理第17列：患者体重
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_18_department(self, series):
        """
        处理第18列：患者住院科室
        编码为数值，按照科室列表编号
        """

        def encode_department(x):
            if pd.isna(x):
                return 0
            x_str = str(x).strip()
            try:
                return self.departments.index(x_str) + 1
            except ValueError:
                return 0

        return series.apply(encode_department)

    def preprocess_column_19_surgery_id(self, series):
        """
        处理第19列：患者手术id
        直接返回
        """
        return series

    def preprocess_column_20_surgery_level(self, series):
        """
        处理第20列：手术等级
        编码为数值：一类=1, 二类=2, 三类=3, 四类=4
        """

        def encode_level(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "一" in x_str or x_str == "1" or x_str == "Ⅰ":
                return 1
            elif "二" in x_str or x_str == "2" or x_str == "Ⅱ":
                return 2
            elif "三" in x_str or x_str == "3" or x_str == "Ⅲ":
                return 3
            elif "四" in x_str or x_str == "4" or x_str == "Ⅳ":
                return 4
            else:
                return -1

        return series.apply(encode_level)

    def preprocess_column_21_surgery_notes(self, series):
        """
        处理第21列：手术注意事项
        保留原始文本，用于文本分析
        """
        return series.fillna("")

    def preprocess_column_22_main_diagnosis(self, series):
        """
        处理第22列：主要诊断
        保留原始文本，用于文本分析，多个诊断用^分隔
        """
        return series.fillna("")

    def preprocess_column_23_planned_surgery_name(self, series):
        """
        处理第23列：拟行手术名称
        保留原始文本，用于文本分析
        """
        return series.fillna("")

    def preprocess_column_24_surgeon(self, series):
        """
        处理第24列：手术医生
        编码为数值，支持多个医生（逗号分隔）
        """
        # 构建医生编码字典
        unique_surgeons = set()
        for item in series.dropna():
            if str(item).strip():
                surgeons = str(item).split(",")
                for surgeon in surgeons:
                    surgeon = surgeon.strip()
                    if surgeon:
                        unique_surgeons.add(surgeon)

        # 为每个医生分配编码
        for i, surgeon in enumerate(sorted(unique_surgeons)):
            self.surgeon_encoding[surgeon] = i + 1

        def encode_surgeons(x):
            if pd.isna(x) or not str(x).strip():
                return [0]  # 缺失值编码为0

            surgeons = str(x).split(",")
            codes = []
            for surgeon in surgeons:
                surgeon = surgeon.strip()
                if surgeon in self.surgeon_encoding:
                    codes.append(self.surgeon_encoding[surgeon])
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_surgeons)

    def preprocess_column_25_assistant(self, series):
        """
        处理第25列：手术助手
        编码为数值，支持多个助手（逗号分隔）
        """

        def encode_assistants(x):
            if pd.isna(x) or not str(x).strip():
                return [0]

            assistants = str(x).split(",")
            codes = []
            for assistant in assistants:
                assistant = assistant.strip()
                # 简化处理，用哈希值编码
                if assistant:
                    codes.append(hash(assistant) % 10000)
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_assistants)

    def preprocess_column_26_anesthesiologist(self, series):
        """
        处理第26列：手术排班的麻醉医生
        编码为数值，支持多个医生（逗号分隔）
        """
        # 构建麻醉医生编码字典
        unique_anesthesiologists = set()
        for item in series.dropna():
            if str(item).strip():
                doctors = str(item).split(",")
                for doctor in doctors:
                    doctor = doctor.strip()
                    if doctor:
                        unique_anesthesiologists.add(doctor)

        # 为每个麻醉医生分配编码
        for i, doctor in enumerate(sorted(unique_anesthesiologists)):
            self.anesthesiologist_encoding[doctor] = i + 1

        def encode_anesthesiologists(x):
            if pd.isna(x) or not str(x).strip():
                return [0]

            doctors = str(x).split(",")
            codes = []
            for doctor in doctors:
                doctor = doctor.strip()
                if doctor in self.anesthesiologist_encoding:
                    codes.append(self.anesthesiologist_encoding[doctor])
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_anesthesiologists)

    def preprocess_column_27_anesthesia_assistant(self, series):
        """
        处理第27列：手术排班的麻醉助手
        编码为数值，支持多个助手（逗号分隔）
        """

        def encode_anesthesia_assistants(x):
            if pd.isna(x) or not str(x).strip():
                return [0]

            assistants = str(x).split(",")
            codes = []
            for assistant in assistants:
                assistant = assistant.strip()
                if assistant:
                    codes.append(hash(assistant) % 10000)
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_anesthesia_assistants)

    def preprocess_column_28_scrub_nurse(self, series):
        """
        处理第28列：手术排班的洗手护士
        编码为数值，支持多个护士（逗号分隔）
        """

        def encode_nurses(x):
            if pd.isna(x) or not str(x).strip():
                return [0]

            nurses = str(x).split(",")
            codes = []
            for nurse in nurses:
                nurse = nurse.strip()
                if nurse:
                    codes.append(hash(nurse) % 10000)
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_nurses)

    def preprocess_column_29_circulating_nurse(self, series):
        """
        处理第29列：手术排班的巡回护士
        编码为数值，支持多个护士（逗号分隔）
        """

        def encode_nurses(x):
            if pd.isna(x) or not str(x).strip():
                return [0]

            nurses = str(x).split(",")
            codes = []
            for nurse in nurses:
                nurse = nurse.strip()
                if nurse:
                    codes.append(hash(nurse) % 10000)
                else:
                    codes.append(0)
            return codes

        return series.apply(encode_nurses)

    def preprocess_column_30_surgery_type(self, series):
        """
        处理第30列：患者详情中对应的手术类型
        编码：择期手术=1, 急诊手术=2, 其他=0
        """

        def encode_surgery_type(x):
            if pd.isna(x):
                return 0
            x_str = str(x).strip()
            if "择期" in x_str:
                return 1
            elif "急诊" in x_str:
                return 2
            else:
                return 0

        return series.apply(encode_surgery_type)

    def preprocess_column_31_room_id(self, series):
        """
        处理第31列：手术排班的手术房间id
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_32_schedule_case_number(self, series):
        """
        处理第32列：手术排班的排班台次
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_33_anesthesia_method(self, series):
        """
        处理第33列：手术排班的麻醉方法
        保留原始文本，用于文本分析
        """
        return series.fillna("")

    def preprocess_column_34_scheduled_surgery_time(self, series):
        """
        处理第34列：手术排班的手术时间
        转换为datetime格式
        """
        return pd.to_datetime(series, errors="coerce")

    def preprocess_column_35_respiratory_history(self, series):
        """
        处理第35列：麻醉前访视呼吸系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_36_endocrine_history(self, series):
        """
        处理第36列：麻醉前访视内分泌系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_37_circulatory_history(self, series):
        """
        处理第37列：麻醉前访视循环系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_38_digestive_history(self, series):
        """
        处理第38列：麻醉前访视消化系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_39_nervous_history(self, series):
        """
        处理第39列：麻醉前访视神经系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_40_immune_history(self, series):
        """
        处理第40列：麻醉前访视免疫系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_41_spine_limbs_history(self, series):
        """
        处理第41列：麻醉前访视脊柱四肢病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_42_urinary_history(self, series):
        """
        处理第42列：麻醉前访视泌尿系统病史
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_43_treatment_history(self, series):
        """
        处理第43列：麻醉前访视病史治疗情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_44_allergy_flag(self, series):
        """
        处理第44列：麻醉前访视-是否存在过敏史
        编码为布尔值：有=True, 无=False
        """

        def encode_allergy_flag(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip()
            return x_str in ["有", "是", "true", "True", "1"]

        return series.apply(encode_allergy_flag)

    def preprocess_column_45_allergy_details(self, series):
        """
        处理第45列：麻醉前访视-过敏史详情
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_46_smoking_flag(self, series):
        """
        处理第46列：麻醉前访视吸烟史标识
        编码为布尔值：有=True, 无=False
        """

        def encode_smoking_flag(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip()
            return x_str in ["有", "是", "true", "True", "1"]

        return series.apply(encode_smoking_flag)

    def preprocess_column_47_smoking_duration(self, series):
        """
        处理第47列：麻醉前访视吸烟史时长
        转换为数值，支持年、月等时间单位，处理各种描述性表达
        """

        def extract_smoking_duration(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip()

            # 处理空值或表示无吸烟史的情况
            if not x_str or x_str in ["/", "-", "无", "没有"]:
                return 0  # 无吸烟史

            # 处理描述性词汇
            if x_str in ["不详", "不明", "未知", "不清"]:
                return -2  # 不详
            elif x_str in ["偶有", "偶尔", "少量", "偶然"]:
                return 0.5  # 偶尔吸烟，编码为0.5年
            elif x_str in ["多", "很多", "数", "数十", "几十"]:
                return 25  # 用25年表示"很多年"
            elif "三无人员" in x_str:
                return 0  # 三无人员视为无吸烟史

            # 处理已戒烟的情况
            if "已戒" in x_str or "戒烟" in x_str:
                # 提取戒烟时间，但返回原吸烟时长的估计值
                match = re.search(r"已戒(\d+)", x_str)
                if match:
                    # 如果戒烟时间明确，可以设为负值表示已戒烟
                    return -float(match.group(1))  # 用负数表示已戒烟多少年
                else:
                    return -10  # 已戒烟但时间不明，用-10表示

            # 尝试直接转换为数值
            try:
                num = float(x_str)
                return num
            except ValueError:
                pass

            # 处理范围表达（如15-20, 3-4）
            range_pattern = r"(\d+)\s*[-~]\s*(\d+)"
            range_match = re.search(range_pattern, x_str)
            if range_match:
                start = float(range_match.group(1))
                end = float(range_match.group(2))
                return (start + end) / 2  # 返回范围中间值

            # 处理带+号的表达（如30+, 10+, 20+）
            plus_pattern = r"(\d+(?:\.\d+)?)\s*\+"
            plus_match = re.search(plus_pattern, x_str)
            if plus_match:
                value = float(plus_match.group(1))
                return value + 5  # 加5年作为估计

            # 处理"余"字表达（如20余, 30余, 40余）
            yu_pattern = r"(\d+(?:\.\d+)?)\s*余"
            yu_match = re.search(yu_pattern, x_str)
            if yu_match:
                return float(yu_match.group(1)) + 2  # 加2年作为估计

            # 处理"多"字表达（如10多, 50多）
            duo_pattern = r"(\d+(?:\.\d+)?)\s*多"
            duo_match = re.search(duo_pattern, x_str)
            if duo_match:
                return float(duo_match.group(1)) + 3  # 加3年作为估计

            # 处理一般数值和单位
            pattern = r"(\d+(?:\.\d+)?)\s*(年|月|天|日)?"
            match = re.search(pattern, x_str)

            if match:
                value = float(match.group(1))
                unit = match.group(2)

                # 根据单位转换为年
                if unit == "月":
                    return round(value / 12, 2)
                elif unit in ["天", "日"]:
                    return round(value / 365, 3)
                else:  # 默认为年
                    return value

            # 如果无法解析，返回-2表示不详
            return -2

        return series.apply(extract_smoking_duration)

    def preprocess_column_48_alcoholism_flag(self, series):
        """
        处理第48列：麻醉前访视酗酒史
        编码为布尔值：有=True, 无=False
        """

        def encode_alcoholism_flag(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip()
            return x_str in ["有", "是", "true", "True", "1"]

        return series.apply(encode_alcoholism_flag)

    def preprocess_column_49_alcoholism_duration(self, series):
        """
        处理第49列：麻醉前访视酗酒史时长
        转换为数值，支持年、月等时间单位
        """

        def extract_alcoholism_duration(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip()

            # 处理空值或表示无酗酒史的情况
            if not x_str or x_str in ["/", "-", "无", "没有"]:
                return 0  # 无酗酒史

            # 处理描述性词汇
            if x_str in ["不详", "不明", "未知"]:
                return -2  # 不详
            elif x_str in ["偶有", "偶尔", "少量", "偶然"]:
                return 0.5  # 偶尔饮酒，编码为0.5年

            # 尝试直接转换为数值
            try:
                num = float(x_str)
                return num
            except ValueError:
                pass

            # 使用正则表达式提取数值和单位
            import re

            pattern = r"(\d+(?:\.\d+)?)\s*(年|月|天|日)?"
            match = re.search(pattern, x_str)

            if match:
                value = float(match.group(1))
                unit = match.group(2)

                # 根据单位转换为年
                if unit == "月":
                    return round(value / 12, 2)
                elif unit in ["天", "日"]:
                    return round(value / 365, 3)
                else:  # 默认为年
                    return value

            # 如果无法解析，返回-2表示不详
            return -2

        return series.apply(extract_alcoholism_duration)

    def preprocess_column_50_surgery_history_flag(self, series):
        """
        处理第50列：麻醉前访视-是否存在手术史
        编码为布尔值：有=True, 无=False
        """

        def encode_surgery_history_flag(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip()
            return x_str in ["有", "是", "true", "True", "1"]

        return series.apply(encode_surgery_history_flag)

    def preprocess_column_51_surgery_history_details(self, series):
        """
        处理第51列：麻醉前访视-手术史详情
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_52_anesthesia_history(self, series):
        """
        处理第52列：麻醉前访视麻醉史
        将麻醉史转换为编码列表，支持组合麻醉（^分隔）
        """

        def encode_anesthesia_history(x):
            if pd.isna(x):
                return [-1]  # 缺失值

            x_str = str(x).strip()

            # 处理无麻醉史的情况
            if x_str in ["无", "没有", "无史", "/", "-"]:
                return [0]  # 无麻醉史

            # 处理不清楚的情况
            if x_str in ["不清", "不明", "未知", "不详"]:
                return [-2]  # 不详

            # 定义麻醉类型编码
            anesthesia_codes = {
                "局麻": 1,
                "脊麻": 2,
                "全麻": 3,
            }

            # 处理组合麻醉（用^分隔）
            if "^" in x_str:
                parts = x_str.split("^")
                code_list = []

                for part in parts:
                    part = part.strip()
                    if part == "不清":
                        code_list.append(-2)
                    elif part in anesthesia_codes:
                        code_list.append(anesthesia_codes[part])
                    else:
                        code_list.append(-2)  # 未知类型标记为不详

                return code_list

            # 处理单一麻醉类型
            elif x_str in anesthesia_codes:
                return [anesthesia_codes[x_str]]

            # 其他未分类情况
            else:
                return [-2]  # 标记为不详

        return series.apply(encode_anesthesia_history)

    def preprocess_column_53_current_medications(self, series):
        """
        处理第53列：麻醉前访视现用药物
        将药物类型转换为编码列表，支持组合药物（^分隔）
        """

        def encode_current_medications(x):
            if pd.isna(x):
                return [-1]  # 缺失值

            x_str = str(x).strip()

            # 处理无药物的情况
            if x_str in ["无", "没有", "无用药", "/", "-"]:
                return [0]  # 无药物

            # 定义药物类型编码
            medication_codes = {
                "降压类": 1,
                "抗生素": 2,
                "支气管扩张类": 3,
                "激素类": 4,
                "抗凝类": 5,
                "othermedicine": 6,  # 其他药物
            }

            # 处理组合药物（用^分隔）
            if "^" in x_str:
                parts = x_str.split("^")
                code_list = []

                for part in parts:
                    part = part.strip()
                    if part in medication_codes:
                        code_list.append(medication_codes[part])
                    else:
                        # 未知药物类型用6表示（othermedicine）
                        code_list.append(6)

                return code_list

            # 处理单一药物类型
            elif x_str in medication_codes:
                return [medication_codes[x_str]]

            # 其他未分类情况
            else:
                return [6]  # 标记为其他药物

        return series.apply(encode_current_medications)

    def preprocess_column_54_mallampati_grade(self, series):
        """
        处理第54列：麻醉前访视查体-Mallampati气道分级
        转换为数值：1, 2, 3, 4级，缺失值=-1
        """

        def encode_mallampati(x):
            if pd.isna(x):
                return -1
            try:
                return int(float(str(x).strip()))
            except:
                return -1

        return series.apply(encode_mallampati)

    def preprocess_column_55_cardiopulmonary_auscultation(self, series):
        """
        处理第55列：麻醉前访视查体-心肺听诊
        编码：正常=1, 异常=0, 缺失=-1
        """

        def encode_auscultation(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 1
            elif "异常" in x_str:
                return 0
            else:
                return -1

        return series.apply(encode_auscultation)

    def preprocess_column_56_auscultation_details(self, series):
        """
        处理第56列：麻醉前访视查体-心肺听诊详情
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_57_wbc_count(self, series):
        """
        处理第57列：术前检查中血常规的白细胞计数
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_58_rbc_count(self, series):
        """
        处理第58列：术前检查中血常规的红细胞计数
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_59_hemoglobin(self, series):
        """
        处理第59列：术前检查中血常规的血红蛋白含量
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_60_platelet_count(self, series):
        """
        处理第60列：术前检查中血常规的血小板计数
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_61_liver_function_abnormal(self, series):
        """
        处理第61列：术前检查中肝功能是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_liver_function(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_liver_function)

    def preprocess_column_62_liver_function_description(self, series):
        """
        处理第62列：术前检查中肝功能异常描述
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_63_kidney_function_abnormal(self, series):
        """
        处理第63列：术前检查中肾功能是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_kidney_function(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_kidney_function)

    def preprocess_column_64_kidney_function_description(self, series):
        """
        处理第64列：术前检查中肾功能异常描述
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_65_electrolyte_abnormal(self, series):
        """
        处理第65列：术前检查中电解质是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_electrolyte(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_electrolyte)

    def preprocess_column_66_electrolyte_description(self, series):
        """
        处理第66列：术前检查中电解质异常描述
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_67_ecg_abnormal(self, series):
        """
        处理第67列：术前检查中心电图检查是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_ecg(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_ecg)

    def preprocess_column_68_ecg_details(self, series):
        """
        处理第68列：术前检查中心电图检查详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_69_coagulation_abnormal(self, series):
        """
        处理第69列：术前检查中凝血筛选是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_coagulation(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_coagulation)

    def preprocess_column_70_coagulation_details(self, series):
        """
        处理第70列：术前检查中凝血筛选详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_71_preop_eight_abnormal(self, series):
        """
        处理第71列：术前检查中术前八项是否存在异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_preop_eight(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_preop_eight)

    def preprocess_column_72_preop_eight_details(self, series):
        """
        处理第72列：术前检查中术前八项是详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_73_blood_gas_abnormal(self, series):
        """
        处理第73列：术前检查中血气是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_blood_gas(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_blood_gas)

    def preprocess_column_74_blood_gas_details(self, series):
        """
        处理第74列：术前检查中血气详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_75_lung_function_abnormal(self, series):
        """
        处理第75列：术前检查中肺功能是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_lung_function(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_lung_function)

    def preprocess_column_76_lung_function_details(self, series):
        """
        处理第76列：术前检查中肺功能详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_77_radiology_results(self, series):
        """
        处理第77列：术前检查中放射结果详情描述
        保留原始文本，用于文本分析
        """
        return series.fillna("")

    def preprocess_column_78_echo_abnormal(self, series):
        """
        处理第78列：术前检查中超声心动图是否异常
        编码：正常=0, 异常=1, 缺失=-1
        """

        def encode_echo(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if "正常" in x_str:
                return 0
            elif "异常" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_echo)

    def preprocess_column_79_echo_details(self, series):
        """
        处理第79列：术前检查超声心动图异常详细情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_80_asa_grade(self, series):
        """
        处理第80列：总体评估时的ASA分级
        编码为数值：Ⅰ=1, Ⅱ=2, Ⅲ=3, Ⅳ=4, Ⅴ=5
        """

        def encode_asa(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if x_str == "Ⅰ":
                return 1
            elif x_str == "Ⅱ":
                return 2
            elif x_str == "Ⅲ":
                return 3
            elif x_str == "Ⅳ":
                return 4
            elif x_str == "Ⅴ":
                return 5
            else:
                return -1

        return series.apply(encode_asa)

    def preprocess_column_81_emergency_surgery_flag(self, series):
        """
        处理第81列：总体评估时的急诊手术标识
        编码：E=1, 其他=0, 缺失=-1
        """

        def encode_emergency(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if x_str == "E":
                return 1
            else:
                return 0

        return series.apply(encode_emergency)

    def preprocess_column_82_special_situation(self, series):
        """
        处理第82列：麻醉前访视中的特殊情况记录
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_83_other_system_abnormal(self, series):
        """
        处理第83列：麻醉前访视其它系统病史是否异常
        编码为布尔值：有=True, 无=False
        """

        def encode_other_system_abnormal(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip()
            return x_str in ["有", "是", "true", "True", "1"]

        return series.apply(encode_other_system_abnormal)

    def preprocess_column_84_other_system_details(self, series):
        """
        处理第84列：麻醉前访视其它系统病史异常情况
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_85_other_system_names(self, series):
        """
        处理第85列：其它麻醉前访视其它系统病史名称
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_86_general_condition(self, series):
        """
        处理第86列：麻醉前访视查体-一般情况
        转换为数值编码
        """

        def encode_general_condition(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip().lower()

            # 良好状态
            if any(word in x_str for word in ["良好", "好", "good", "正常", "normal"]):
                return 3

            # 一般状态
            elif any(word in x_str for word in ["一般", "fair", "average", "尚可"]):
                return 2

            # 较差状态
            elif any(word in x_str for word in ["差", "poor", "不良", "虚弱", "weak"]):
                return 1

            # 危重状态
            elif any(word in x_str for word in ["危重", "critical", "严重", "severe"]):
                return 0

            # 其他未分类情况
            else:
                return -1

        return series.apply(encode_general_condition)

    def preprocess_column_87_mouth_opening(self, series):
        """
        处理第87列：麻醉前访视查体-开口度
        转换为数值，缺失值填充为-1
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def preprocess_column_88_mouth_opening_degree(self, series):
        """
        处理第88列：麻醉前访视查体-张口度
        编码：good=3, middle=2, poor=1, 缺失=-1
        """

        def encode_mouth_opening_degree(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip().lower()
            if "good" in x_str or "好" in x_str:
                return 3
            elif "middle" in x_str or "中" in x_str:
                return 2
            elif "poor" in x_str or "差" in x_str:
                return 1
            else:
                return -1

        return series.apply(encode_mouth_opening_degree)

    def preprocess_column_89_heart_function_grade(self, series):
        """
        处理第89列：麻醉计划风险评估时患者心功能分级(New York)
        转换为数值：Ⅰ=1, Ⅱ=2, Ⅲ=3, Ⅳ=4
        """

        def encode_heart_function(x):
            if pd.isna(x):
                return -1
            x_str = str(x).strip()
            if x_str == "Ⅰ":
                return 1
            elif x_str == "Ⅱ":
                return 2
            elif x_str == "Ⅲ":
                return 3
            elif x_str == "Ⅳ":
                return 4
            else:
                try:
                    return int(float(x_str))
                except:
                    return -1

        return series.apply(encode_heart_function)

    def preprocess_column_90_anesthesia_surgery_risk(self, series):
        """
        处理第90列：麻醉计划风险评估时患者麻醉手术风险类别
        转换为数值编码
        """

        def encode_risk_category(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip().lower()

            # 根据实际数据进行分类编码
            if "category1" in x_str or "类别1" in x_str:
                return 1  # 最低风险
            elif "category2" in x_str or "类别2" in x_str:
                return 2  # 低风险
            elif "category3" in x_str or "类别3" in x_str:
                return 3  # 中等风险
            elif "category4" in x_str or "类别4" in x_str:
                return 4  # 高风险
            elif "category5" in x_str or "类别5" in x_str:
                return 5  # 极高风险
            elif "normal" in x_str or "正常" in x_str:
                return 0  # 正常无风险
            elif "low" in x_str or "低" in x_str:
                return 1  # 低风险
            elif "medium" in x_str or "中" in x_str:
                return 2  # 中等风险
            elif "high" in x_str or "高" in x_str:
                return 3  # 高风险
            else:
                # 对于未知类别，可以用哈希值映射到一个固定范围
                return hash(str(x).strip()) % 10

        return series.apply(encode_risk_category)

    def preprocess_column_91_planned_anesthesia_method(self, series):
        """
        处理第91列：拟行麻醉方式
        保留原始文本，支持多种麻醉方式（^分隔）
        """
        return series.fillna("")

    def preprocess_column_92_alternative_anesthesia_method(self, series):
        """
        处理第92列：备选麻醉方式
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_93_planned_anesthetics(self, series):
        """
        处理第93列：拟选用麻醉药
        保留原始文本，支持多种药物（^分隔）
        """
        return series.fillna("")

    def preprocess_column_94_monitoring_items(self, series):
        """
        处理第94列：拟监测项目
        保留原始文本，支持多个监测项目（^分隔）
        """
        return series.fillna("")

    def preprocess_column_95_potential_risks(self, series):
        """
        处理第95列：术中可能出现的风险
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_96_anesthesia_indications(self, series):
        """
        处理第96列：麻醉计划风险评估时患者有无麻醉适应症标识
        编码为布尔值
        """

        def encode_indications(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip().lower()
            return x_str in ["true", "1", "yes", "是", "t"]

        return series.apply(encode_indications)

    def preprocess_column_97_surgery_anesthesia_ready(self, series):
        """
        处理第97列：麻醉计划风险评估时患者能否按期进行手术麻醉标识
        编码为布尔值
        """

        def encode_ready(x):
            if pd.isna(x):
                return False
            x_str = str(x).strip().lower()
            return x_str in ["true", "1", "yes", "是", "t"]

        return series.apply(encode_ready)

    def preprocess_column_98_delay_reason(self, series):
        """
        处理第98列：麻醉计划风险评估时患者延期麻醉原因
        编码为类别：无延期=0, 有具体原因=1, 缺失=-1
        """

        def encode_delay_reason(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip()

            # 表示无延期原因的各种表达
            no_delay_indicators = [
                "无",
                "/",
                "//",
                "-",
                "未见",
                "未改",
                "未更改",
                "、",
                "没有",
                "正常",
                "按时",
                "无延期",
                "准时",
            ]

            # 如果是空字符串或只包含空白字符
            if not x_str or x_str.isspace():
                return 0  # 视为无延期

            # 检查是否为"无延期"的表达
            if any(indicator in x_str for indicator in no_delay_indicators):
                return 0  # 无延期原因

            # 如果包含具体的延期原因描述
            elif len(x_str) > 2 and not all(
                c in "/-、\\无未见改更正常按时准" for c in x_str
            ):
                return 1  # 有具体延期原因

            else:
                return 0  # 默认为无延期

        return series.apply(encode_delay_reason)

    def preprocess_column_99_postop_unit(self, series):
        """
        处理第99列：术后患者拟转入单元
        编码为数值：ward=1, PACU=2, ICU=3, 缺失=-1
        """

        def encode_postop_unit(x):
            if pd.isna(x):
                return -1  # 缺失值

            x_str = str(x).strip().upper()

            # 病房相关
            if "WARD" in x_str or "病房" in str(x).strip():
                return 1

            # PACU相关
            elif "PACU" in x_str:
                return 2

            # ICU相关
            elif "ICU" in x_str:
                return 3

            # 其他未分类情况
            else:
                return 0

        return series.apply(encode_postop_unit)

    def preprocess_column_100_anesthesiologist_name(self, series):
        """
        处理第100列：麻醉医生姓名
        编码为数值
        """

        def encode_anesthesiologist_name(x):
            if pd.isna(x) or not str(x).strip():
                return 0
            name = str(x).strip()
            return hash(name) % 10000

        return series.apply(encode_anesthesiologist_name)

    def preprocess_column_101_ultrasound_guided_anesthesia(self, series):
        """
        处理第101列：超声引导麻醉名称
        保留原始文本
        """
        return series.fillna("")

    def preprocess_column_102_ultrasound_conclusion(self, series):
        """
        处理第102列：超声检查结论信息
        保留原始文本，用于文本分析
        """
        return series.fillna("")

    def preprocess_column_103_instrument_count_surgery_name(self, series):
        """
        处理第103列：手术器械清点单手术名称
        保留原始文本，支持多个手术名称（^分隔）
        """
        return series.fillna("")

    def preprocess_column_104_surgery_room_duration(self, series):
        """
        处理第104列：手术间时长 - 目标变量
        转换为数值，这是我们要预测的目标
        """
        return pd.to_numeric(series, errors="coerce").fillna(-1)

    def process_all_columns(self, df):
        """
        处理所有列数据
        """
        print("开始处理所有列数据...")

        # 使用字典收集所有处理后的列，避免DataFrame碎片化
        processed_columns = {}
        column_names = df.columns.tolist()

        for i, col_name in enumerate(column_names):
            method_name = f"preprocess_column_{i+1}_{self._get_method_suffix(col_name)}"

            if hasattr(self, method_name):
                print(f"处理第{i+1}列: {col_name}")
                try:
                    processed_series = getattr(self, method_name)(df[col_name])
                    processed_columns[col_name] = processed_series
                except Exception as e:
                    print(f"处理第{i+1}列时出错: {e}")
                    processed_columns[col_name] = df[col_name]  # 保留原始数据
            else:
                print(f"第{i+1}列没有对应的处理方法: {col_name}")
                processed_columns[col_name] = df[col_name]  # 保留原始数据

        # 一次性创建DataFrame，避免性能警告
        processed_df = pd.DataFrame(processed_columns)

        print("所有列处理完成!")
        return processed_df

    def _get_method_suffix(self, col_name):
        """
        根据列名生成方法后缀
        """
        # 简化列名，用于方法名匹配
        name_mapping = {
            "手麻系统中手术状态时间历史记录唯一标识": "unique_id",
            "患者入手术室时间": "room_entry_time",
            "术前准备时间": "prep_time",
            "患者入手术间时间": "surgery_room_entry",
            "麻醉开始时间": "anesthesia_start",
            "手术开始时间": "surgery_start",
            "手术结束时间": "surgery_end",
            "麻醉结束时间": "anesthesia_end",
            "患者出手术间时间": "room_exit_time",
            "手术排班完成台次": "completed_cases",
            "术后时长预测时间": "prediction_time",
            "非计划二次手术标识": "unplanned_second_surgery",
            "院内临床信息患者唯一标识": "patient_id",
            "患者性别": "patient_gender",
            "患者年龄": "patient_age",
            "患者身高": "patient_height",
            "患者体重": "patient_weight",
            "患者住院科室": "department",
            "患者手术id": "surgery_id",
            "手术等级": "surgery_level",
            "手术注意事项": "surgery_notes",
            "主要诊断": "main_diagnosis",
            "拟行手术名称": "planned_surgery_name",
            "手术医生": "surgeon",
            "手术助手": "assistant",
            "手术排班的麻醉医生": "anesthesiologist",
            "手术排班的麻醉助手": "anesthesia_assistant",
            "手术排班的洗手护士": "scrub_nurse",
            "手术排班的巡回护士": "circulating_nurse",
            "患者详情中对应的手术类型": "surgery_type",
            "手术排班的手术房间id": "room_id",
            "手术排班的排班台次": "schedule_case_number",
            "手术排班的麻醉方法": "anesthesia_method",
            "手术排班的手术时间": "scheduled_surgery_time",
            "麻醉前访视呼吸系统病史": "respiratory_history",
            "麻醉前访视内分泌系统病史": "endocrine_history",
            "麻醉前访视循环系统病史": "circulatory_history",
            "麻醉前访视消化系统病史": "digestive_history",
            "麻醉前访视神经系统病史": "nervous_history",
            "麻醉前访视免疫系统病史": "immune_history",
            "麻醉前访视脊柱四肢病史": "spine_limbs_history",
            "麻醉前访视泌尿系统病史": "urinary_history",
            "麻醉前访视病史治疗情况": "treatment_history",
            "麻醉前访视-是否存在过敏史": "allergy_flag",
            "麻醉前访视-过敏史详情": "allergy_details",
            "麻醉前访视吸烟史标识": "smoking_flag",
            "麻醉前访视吸烟史时长": "smoking_duration",
            "麻醉前访视酗酒史": "alcoholism_flag",
            "麻醉前访视酗酒史时长": "alcoholism_duration",
            "麻醉前访视-是否存在手术史": "surgery_history_flag",
            "麻醉前访视-手术史详情": "surgery_history_details",
            "麻醉前访视麻醉史": "anesthesia_history",
            "麻醉前访视现用药物": "current_medications",
            "麻醉前访视查体-Mallampati气道分级": "mallampati_grade",
            "麻醉前访视查体-心肺听诊": "cardiopulmonary_auscultation",
            "麻醉前访视查体-心肺听诊详情": "auscultation_details",
            "术前检查中血常规的白细胞计数": "wbc_count",
            "术前检查中血常规的红细胞计数": "rbc_count",
            "术前检查中血常规的血红蛋白含量": "hemoglobin",
            "术前检查中血常规的血小板计数": "platelet_count",
            "术前检查中肝功能是否异常": "liver_function_abnormal",
            "术前检查中肝功能异常描述": "liver_function_description",
            "术前检查中肾功能是否异常": "kidney_function_abnormal",
            "术前检查中肾功能异常描述": "kidney_function_description",
            "术前检查中电解质是否异常": "electrolyte_abnormal",
            "术前检查中电解质异常描述": "electrolyte_description",
            "术前检查中心电图检查是否异常": "ecg_abnormal",
            "术前检查中心电图检查详细情况": "ecg_details",
            "术前检查中凝血筛选是否异常": "coagulation_abnormal",
            "术前检查中凝血筛选详细情况": "coagulation_details",
            "术前检查中术前八项是否存在异常": "preop_eight_abnormal",
            "术前检查中术前八项是详细情况": "preop_eight_details",
            "术前检查中血气是否异常": "blood_gas_abnormal",
            "术前检查中血气详细情况": "blood_gas_details",
            "术前检查中肺功能是否异常": "lung_function_abnormal",
            "术前检查中肺功能详细情况": "lung_function_details",
            "术前检查中放射结果详情描述": "radiology_results",
            "术前检查中超声心动图是否异常": "echo_abnormal",
            "术前检查超声心动图异常详细情况": "echo_details",
            "总体评估时的ASA分级": "asa_grade",
            "总体评估时的急诊手术标识": "emergency_surgery_flag",
            "麻醉前访视中的特殊情况记录": "special_situation",
            "麻醉前访视其它系统病史是否异常": "other_system_abnormal",
            "麻醉前访视其它系统病史异常情况": "other_system_details",
            "其它麻醉前访视其它系统病史名称": "other_system_names",
            "麻醉前访视查体-一般情况": "general_condition",
            "麻醉前访视查体-开口度": "mouth_opening",
            "麻醉前访视查体-张口度": "mouth_opening_degree",
            "麻醉计划风险评估时患者心功能分级(New York)": "heart_function_grade",
            "麻醉计划风险评估时患者麻醉手术风险类别": "anesthesia_surgery_risk",
            "拟行麻醉方式": "planned_anesthesia_method",
            "备选麻醉方式": "alternative_anesthesia_method",
            "拟选用麻醉药": "planned_anesthetics",
            "拟监测项目": "monitoring_items",
            "术中可能出现的风险": "potential_risks",
            "麻醉计划风险评估时患者有无麻醉适应症标识": "anesthesia_indications",
            "麻醉计划风险评估时患者能否按期进行手术麻醉标识": "surgery_anesthesia_ready",
            "麻醉计划风险评估时患者延期麻醉原因": "delay_reason",
            "术后患者拟转入单元": "postop_unit",
            "麻醉医生姓名": "anesthesiologist_name",
            "超声引导麻醉名称": "ultrasound_guided_anesthesia",
            "超声检查结论信息": "ultrasound_conclusion",
            "手术器械清点单手术名称": "instrument_count_surgery_name",
            "手术间时长": "surgery_room_duration",
        }

        return name_mapping.get(col_name, "unknown")

    def save_processed_data(self, processed_df, output_path):
        """
        保存处理后的数据
        """
        print(f"保存处理后的数据到: {output_path}")
        if output_path.endswith(".pkl"):
            processed_df.to_pickle(output_path)  # 保存为pkl格式
        elif output_path.endswith(".xlsx"):
            processed_df.to_excel(output_path, index=False)
        else:
            processed_df.to_csv(output_path, index=False, encoding="utf-8")
        print("数据保存完成!")

    def save_encodings(self, output_dir):
        """
        保存编码映射
        """
        # 保存医生编码映射
        with open(f"{output_dir}/surgeon_encoding.json", "w", encoding="utf-8") as f:
            json.dump(self.surgeon_encoding, f, ensure_ascii=False, indent=2)

        with open(
            f"{output_dir}/anesthesiologist_encoding.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.anesthesiologist_encoding, f, ensure_ascii=False, indent=2)

        print("编码映射保存完成!")


def main():
    """
    主函数：执行数据预处理流程
    """
    # 初始化预处理器
    preprocessor = SurgeryDataPreprocessor()

        # 数据文件路径
    input_file = r"G:\手术时长预测\op\duration_predict_train\25年6月数据集\val_set.pkl"  # 改为pkl
    output_file = r"G:\手术时长预测\op\duration_predict_train\25年6月数据集\processed_val.pkl"  # 改为pkl
    output_dir = r"G:\手术时长预测\op\duration_predict_train\25年6月数据集"

    # 加载数据
    df = preprocessor.load_data(input_file)

    # 处理所有列
    processed_df = preprocessor.process_all_columns(df)

    # 保存处理后的数据
    preprocessor.save_processed_data(processed_df, output_file)

    # 保存编码映射
    preprocessor.save_encodings(output_dir)

    # 打印数据统计信息
    print("\n数据处理统计:")
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {processed_df.shape}")
    print(f"目标变量(手术间时长)统计:")
    print(processed_df["手术间时长"].describe())


if __name__ == "__main__":
    main()
