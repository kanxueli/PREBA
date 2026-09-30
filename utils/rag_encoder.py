import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
import pickle
import os
import torch
from tqdm import tqdm

class HeterogeneousFeatureEncoder:
    """
    异构特征编码器
    处理手术时长预测数据集中的42个异构特征
    """
    
    def __init__(self, dataset_name='', use_grouped_encoding=False):
        """
        初始化异构特征编码器
        
        Args:
            dataset_name: 数据集名称，用于选择特征配置 ('Multimodal_SDP' 或其他)
            use_grouped_encoding: 是否使用分组编码（True=按权重分组编码BERT特征，False=独立编码每个特征）
        """
        self.dataset_name = dataset_name
        self.use_grouped_encoding = use_grouped_encoding
        
        # 特征类别配置 - 原始数据集
        self.feature_categories = {
            'numerical': ['患者年龄', '手术排班的排班台次'],  # 连续数值特征 2个
            # 'numerical': ['患者年龄', '手术排班的排班台次', '手术间时长'],  # 连续数值特征 3个 # 新增手术间时长特征,用于构建因果一致性数据集
            'ordinal': ['总体评估时的ASA分级', '手术等级', '麻醉计划风险评估时患者心功能分级(New York)'],  # 序数特征 3个
            'categorical_low': ['患者性别', '患者详情中对应的手术类型'],  # 低基数分类特征 2个
            'categorical_high': ['手术医生', '麻醉医生姓名', '手术排班的手术房间id', '手术排班的洗手护士', '手术排班的巡回护士'],  # 高基数分类特征 5个
            'boolean': ['术前检查中肺功能是否异常', '术前检查中血气是否异常', '术前检查中超声心动图是否异常', 
                       '麻醉前访视吸烟史标识', '麻醉前访视酗酒史'],  # 布尔特征 5个
            'text': ['拟行手术名称', '主要诊断', '患者住院科室', 
                    '手术器械清点单手术名称', '手术排班的麻醉方法', '拟选用麻醉药', '手术注意事项', 
                    '术前检查中放射结果详情描述', '术前检查中心电图检查详细情况', '术前检查中凝血筛选详细情况',
                    '麻醉前访视呼吸系统病史', '麻醉前访视神经系统病史', '麻醉前访视消化系统病史',
                    '麻醉前访视脊柱四肢病史', '麻醉前访视泌尿系统病史', '麻醉前访视免疫系统病史',
                    '麻醉前访视其它系统病史异常情况', '麻醉前访视-过敏史详情', '麻醉前访视查体-一般情况',
                    '麻醉前访视查体-心肺听诊详情', '麻醉前访视-手术史详情', '麻醉前访视病史治疗情况',
                    '麻醉前访视麻醉史', '麻醉前访视中的特殊情况记录','患者入手术室时间'],  # 文本特征 25个
        }

        # # 特征类别配置 - MMSDP数据集 (初版数据集，没有测试/验证集)
        # self.feature_categories_MMSDP = {
        #     'numerical': ['patient_age', 'patient_height', 'patient_weight', 'diagnostic_CBC_RBC', 'diagnostic_CBC_WBC', 'diagnostic_CBC_HGB', 'diagnostic_CBC_PLT'],  
        #     # 'numerical': ['patient_age', 'patient_height', 'patient_weight', 'diagnostic_CBC_RBC', 'diagnostic_CBC_WBC', 'diagnostic_CBC_HGB', 'diagnostic_CBC_PLT', 'duration'], # 新增手术间时长特征,用于构建因果一致性数据集
        #     'ordinal': ['level', 'patient_asa'],  
        #     'categorical_low': ['patient_gender', 'type', 'degree_of_mouth_opening', 'neck_activity'], 
        #     'categorical_high': ['surgeon', 'anesthesiologist'], 
        #     'boolean': ['diagnostic_chest', 'diagnostic_ECG', 'diagnostic_liver', 'diagnostic_kidney', 'disease_smoking', 'disease_alcoholism', 'disease_allergies', 'disease_respiratory', 'disease_nerve', 'disease_circulatory'],
        #     'text': ['name', 'anesthesia', 'department', 'patient_diagnose', 'position', 'ICD'],  # 6个文本特征
        # }

        # 特征类别配置 - MMSDP数据集 (2026重新拿的数据集)
        self.feature_categories_MMSDP = {
            'numerical': ['patient_age', 'patient_height', 'patient_weight', 'diagnostic_CBC_RBC', 'diagnostic_CBC_WBC', 'diagnostic_CBC_HGB', 'diagnostic_CBC_PLT'],  
            # 'numerical': ['patient_age', 'patient_height', 'patient_weight', 'diagnostic_CBC_RBC', 'diagnostic_CBC_WBC', 'diagnostic_CBC_HGB', 'diagnostic_CBC_PLT', 'duration'], # 新增手术间时长特征,用于构建因果一致性数据集
            'ordinal': ['level', 'assessment_asa'],  
            'categorical_low': ['patient_gender', 'type', 'degree_of_mouth_opening', 'neck_activity'], 
            'categorical_high': ['surgeons', 'anesthesiologists'], 
            'boolean': ['diagnostic_electrolyte', 'diagnostic_ECG', 'diagnostic_liver', 'diagnostic_kidney', 'disease_smoking', 'disease_alcoholism', 'disease_allergies', 'disease_respiratory', 'disease_nerve', 'disease_circulatory'],
            'text': ['procedures', 'department', 'diagnoses', 'position', 'ICD'],  # 6个文本特征 'anesthesia', 
        }

        # 特征类别配置 - INSPIRE 数据集（id 已由 op_id 改为 id）
        self.feature_categories_INSPIRE = {
            'numerical': ['patient_age', 'patient_weight', 'patient_height'],
            # 'numerical': ['patient_age', 'patient_weight', 'patient_height', 'duration'], # 新增手术间时长特征,用于构建因果一致性数据集
            'ordinal': ['assessment_asa'],
            'categorical_low': ['patient_gender', 'anesthesia'],
            'categorical_high': [],
            'boolean': ['emop'],
            'text': ['department.1', 'ICD', 'diagnoses_text', 'preop_labs_latest_json'],
        }

        # 特征类别配置 - MOVER_EPIC 数据集（英文；暂不使用 patient_weight）
        self.feature_categories_MOVER_EPIC = {
            'numerical': ['patient_age', 'patient_height'],
            # 'numerical': ['patient_age', 'patient_height', 'duration'], # 新增手术间时长特征,用于构建因果一致性数据集
            'ordinal': ['assessment_asa'],
            'categorical_low': ['patient_gender'],
            'categorical_high': [],
            'boolean': [],
            'text': ['procedure_name', 'diagnosis_names'],
        }

        # 特征类别配置 - MOVER_SIS 数据集（英文）
        self.feature_categories_MOVER_SIS = {
            'numerical': ['patient_age', 'patient_height', 'patient_weight'],
            # 'numerical': ['patient_age', 'patient_height', 'patient_weight', 'duration'], # 新增手术间时长特征,用于构建因果一致性数据集
            'ordinal': [],
            'categorical_low': ['patient_gender'],
            'categorical_high': [],
            'boolean': [],
            'text': ['procedure_name', 'preop_medications'],
        }

        # 根据数据集类型选择特征配置
        if dataset_name == 'Multimodal_SDP':
            self.active_feature_categories = self.feature_categories_MMSDP
        elif dataset_name == 'INSPIRE':
            self.active_feature_categories = self.feature_categories_INSPIRE
        elif dataset_name == 'MOVER_EPIC':
            self.active_feature_categories = self.feature_categories_MOVER_EPIC
        elif dataset_name == 'MOVER_SIS':
            self.active_feature_categories = self.feature_categories_MOVER_SIS
        else:
            self.active_feature_categories = self.feature_categories


        
        # 编码器初始化
        self.encoders = self._init_encoders()
        self.is_fitted = False
        self._transform_info_printed = True
        # 记录 BERT 高维特征按权重分组后的信息（跨 text / categorical_high 统一分组）
        # 结构示例：
        # [
        #   {'weight': 0.30, 'features': ['拟行手术名称', '主要诊断', '手术器械清点单手术名称']},
        #   {'weight': 0.15, 'features': [...]},
        # ]
        self.bert_group_info = []
        
        self.feature_categories = self.active_feature_categories
    
    @staticmethod
    def _to_scalar(x):
        """将 list/array 转为标量，避免编码时触发 "truth value of an array is ambiguous"。"""
        if x is None:
            return None
        while isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return '无'
            x = x[0]
        return x
    
    def _ensure_scalar_cells(self, df):
        """确保 DataFrame 中所有 list/array 单元格转为标量，检索单条时常见。"""
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda x: self._to_scalar(x) if isinstance(x, (list, tuple, np.ndarray)) else x)
        return out
    
    def _normalize_boolean_values(self, series):
        """
        标准化boolean特征值，统一处理不同数据集的格式
        
        Args:
            series: pandas Series，包含boolean特征值
        
        Returns:
            numpy array: 标准化后的boolean值数组
        """
        # 填充空值：根据数据集类型选择默认值
        if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
            # MMSDP/INSPIRE：使用 'false' 作为默认值
            filled_series = series.fillna('false')
        # else:
        #     # 原始数据集：使用'否'作为默认值
        #     filled_series = series.fillna('否')
        
        # 转换为字符串
        str_series = filled_series.astype(str)
        
        # 标准化boolean值：统一转换为小写，并处理常见变体
        normalized = []
        for val in str_series:
            val_lower = val.lower().strip()
            # 处理MMSDP数据集的"false"/"true"
            if val_lower in ['false', 'f', '0', '否', '无']:
                normalized.append('false')
            elif val_lower in ['true', 't', '1', '是']:
                normalized.append('true')
            else:
                # 未知值，根据数据集类型使用默认值
                if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                    normalized.append('false')
                else:
                    normalized.append('否')
        
        return np.array(normalized)
    
    def _init_encoders(self):
        """初始化各种编码器"""
        encoders = {
            'numerical': StandardScaler(),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            'categorical_low': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'boolean': LabelEncoder()
        }
        
        # 检查GPU可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # # 如果使用GPU，显示显存信息
        # if torch.cuda.is_available():
        #     gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        #     print(f"GPU显存总量: {gpu_memory:.1f} GB")
        #     print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # 使用bert-base-chinese模型
        try:
            from transformers import AutoTokenizer, AutoModel, BertModel
            print("加载 BERT 模型...")
            # 英文数据集用 bert-base-uncased，其余用中文；优先本地路径
            if self.dataset_name in ('INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                # local_model_path = '/home/data2/LLM_benchmarking/bert-base-uncased'
                # model_name = 'bert-base-uncased'
                # local_model_path = '/home/data2/LLM_benchmarking/ModernBERT-base'
                # model_name = 'ModernBERT-base' # 效果不佳
                # local_model_path = '/home/data2/LLM_benchmarking/medBERT-base'
                # model_name = 'medBERT-base' 
                local_model_path = '/home/data2/LLM_benchmarking/Clinical_ModernBERT'
                model_name = 'Clinical_ModernBERT' 
            else:
                local_model_path = '/home/data2/LLM_benchmarking/bert-base-chinese'
                model_name = 'bert-base-chinese'
            if os.path.exists(local_model_path):
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                # Clinical_ModernBERT 检查点中不包含 pooler 权重，直接用 BertModel 且关闭 pooling 层，避免多余的未初始化权重警告
                if model_name == 'Clinical_ModernBERT':
                    model = BertModel.from_pretrained(local_model_path, add_pooling_layer=False)
                else:
                    model = AutoModel.from_pretrained(local_model_path)
                print(f"成功加载本地模型: {local_model_path}")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                print(f"成功下载并加载模型: {model_name}")
            
            # 将模型移到GPU（如果可用）
            model = model.to(self.device)
            
            # 创建BERT编码器类
            class BERTEncoder:
                def __init__(self, tokenizer, model, device):
                    self.tokenizer = tokenizer
                    self.model = model
                    self.device = device
                    self.empty_vector = np.zeros(768)  # BERT-base的向量维度
                
                def encode(self, texts, batch_size=256):
                    """编码文本为向量，支持批处理"""
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # 过滤空值文本，直接返回零向量
                    valid_texts = []
                    valid_indices = []
                    for i, text in enumerate(texts):
                        if pd.isna(text) or str(text).strip() in ['无', '', 'nan', 'None']:
                            valid_texts.append('[PAD]')  # 占位符
                        else:
                            valid_texts.append(str(text))
                        valid_indices.append(i)
                    
                    if not valid_texts:
                        return np.zeros((len(texts), 768))
                    
                    # 批处理编码
                    all_embeddings = []
                    for i in range(0, len(valid_texts), batch_size):
                        batch_texts = valid_texts[i:i+batch_size]
                        
                        # 对文本进行tokenization
                        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                              max_length=512, return_tensors='pt')  # 减少max_length
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # 获取模型输出
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # 使用[CLS] token的表示作为句子向量
                            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        
                        all_embeddings.append(batch_embeddings)
                        
                        # 清理GPU缓存
                        del inputs, outputs, batch_embeddings
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # 合并所有批次的嵌入
                    embeddings = np.vstack(all_embeddings)
                    
                    # 处理空值：将占位符的向量设为零向量
                    for i, text in enumerate(texts):
                        if pd.isna(text) or str(text).strip() in ['无', '', 'nan', 'None']:
                            embeddings[i] = self.empty_vector
                    
                    return embeddings
            
            # 创建编码器实例
            bert_encoder = BERTEncoder(tokenizer, model, self.device)
            encoders['categorical_high'] = bert_encoder
            encoders['text'] = bert_encoder
                
        except Exception as e:
            print(f"⚠ bert-base-chinese模型加载失败: {e}")
            print("使用TF-IDF备用编码方案...")
            # 使用TF-IDF作为备用方案
            from sklearn.feature_extraction.text import TfidfVectorizer
            encoders['categorical_high'] = TfidfVectorizer(max_features=100, stop_words=None)
            encoders['text'] = TfidfVectorizer(max_features=100, stop_words=None)
        return encoders
    
    def fit(self, df):
        """Fitting heterogeneous feature encoder..."""
        print("Fitting heterogeneous feature encoder...")
        
        # 使用active_feature_categories
        active_categories = self.active_feature_categories
        
        # 创建进度条
        categories = [cat for cat in active_categories.keys() 
                     if any(f in df.columns for f in active_categories[cat])]
        
        with tqdm(total=len(categories), desc="Fitting Encoder", unit="类别") as pbar:
            for category, features in active_categories.items():
                # 检查特征是否存在
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                    
                category_data = df[available_features]
                
                try:
                    if category == 'numerical':
                        # 处理所有数值特征
                        numeric_data = pd.to_numeric(category_data.iloc[:, 0], errors='coerce').fillna(0)
                        for i in range(1, category_data.shape[1]):
                            additional_data = pd.to_numeric(category_data.iloc[:, i], errors='coerce').fillna(0)
                            numeric_data = np.column_stack([numeric_data, additional_data])
                        self.encoders[category].fit(numeric_data)
                    elif category == 'ordinal':
                        # 处理所有序数特征
                        filled_data = category_data.fillna('无')
                        self.encoders[category].fit(filled_data)
                    elif category == 'categorical_low':
                        # 处理所有低基数分类特征
                        filled_data = category_data.fillna('无')
                        self.encoders[category].fit(filled_data)
                    elif category == 'boolean':
                        # 注意：不使用外部的fillna，由_normalize_boolean_values统一处理
                        if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                            boolean_data = self._normalize_boolean_values(category_data.iloc[:, 0])
                        else:
                            boolean_data = category_data.iloc[:, 0].astype(str)
                        
                        for i in range(1, category_data.shape[1]):
                            if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                                additional_data = self._normalize_boolean_values(category_data.iloc[:, i])
                            else:
                                additional_data = category_data.iloc[:, i].astype(str)

                            boolean_data = np.concatenate([boolean_data, additional_data])
                        self.encoders[category].fit(boolean_data)
                    # categorical_high 和 text 不需要fit，直接使用预训练模型
                    
                    if pbar:
                        pbar.set_postfix({"类别": category, "特征数": len(available_features)})
                        pbar.update(1)
                except Exception as e:
                    print(f"  {category} 特征编码器训练失败: {e}")
                    if pbar:
                        pbar.update(1)
                    continue
        
        self.is_fitted = True
        print("Fitting heterogeneous feature encoder completed")
    
    def transform(self, df, show_progress=False, feature_weights=None):
        """
        编码特征
        
        Args:
            df: pandas DataFrame，待编码的数据
            show_progress: 是否显示进度条
            feature_weights: dict，特征名 -> 权重（仅在use_grouped_encoding=True时需要）
        
        Returns:
            dict: {category: np.ndarray}，编码后的特征
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未fit，请先调用fit方法")
        
        # 将 list/array 单元格转为标量
        df = self._ensure_scalar_cells(df)
        
        # 如果启用分组编码，使用transform_grouped的逻辑
        if self.use_grouped_encoding:
            if feature_weights is None:
                # 如果没有提供权重，尝试从rag_weights获取默认权重
                from .rag_weights import (
                    get_feature_weights,
                    get_feature_weights_MMSDP,
                    get_feature_weights_INSPIRE,
                    get_feature_weights_MOVER_EPIC,
                    get_feature_weights_MOVER_SIS,
                )
                if self.dataset_name == 'Multimodal_SDP':
                    feature_weights = get_feature_weights_MMSDP()
                elif self.dataset_name == 'INSPIRE':
                    feature_weights = get_feature_weights_INSPIRE()
                elif self.dataset_name == 'MOVER_EPIC':
                    feature_weights = get_feature_weights_MOVER_EPIC()
                elif self.dataset_name == 'MOVER_SIS':
                    feature_weights = get_feature_weights_MOVER_SIS()
                else:
                    feature_weights = get_feature_weights()
            return self.transform_grouped(df, feature_weights, show_progress)
        
        # 否则使用原始独立编码方式
        encoded_features = {}
        
        # 使用active_feature_categories
        active_categories = self.active_feature_categories
        
        categories = [cat for cat in active_categories.keys() 
                     if any(f in df.columns for f in active_categories[cat])]
        
        # 创建进度条（仅在需要时显示）
        if show_progress:
            pbar = tqdm(total=len(categories), desc="编码特征", unit="类别")
        else:
            pbar = None
        
        try:
            for category, features in active_categories.items():
                # 检查特征是否存在
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                    
                category_data = df[available_features]
                
                try:
                    if category == 'numerical':
                        # 处理所有数值特征
                        numeric_data = pd.to_numeric(category_data.iloc[:, 0], errors='coerce').fillna(0)
                        for i in range(1, category_data.shape[1]):
                            additional_data = pd.to_numeric(category_data.iloc[:, i], errors='coerce').fillna(0)
                            numeric_data = np.column_stack([numeric_data, additional_data])
                        encoded_features[category] = self.encoders[category].transform(numeric_data)
                        
                        # 打印编码结果
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    elif category == 'ordinal':
                        # 处理所有序数特征
                        filled_data = category_data.fillna('无')
                        encoded_features[category] = self.encoders[category].transform(filled_data)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    elif category == 'categorical_low':
                        # 处理所有低基数分类特征
                        filled_data = category_data.fillna('无')
                        encoded_features[category] = self.encoders[category].transform(filled_data)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    elif category == 'boolean':
                        # 处理所有布尔特征 - 需要分别处理每个特征
                        boolean_encoded = []
                        for i in range(category_data.shape[1]):
                            # 使用与fit阶段相同的标准化方法
                            if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                                single_boolean = self._normalize_boolean_values(category_data.iloc[:, i])
                            else:
                                single_boolean = category_data.iloc[:, i].fillna('否').astype(str)
                                
                            encoded_single = self.encoders[category].transform(single_boolean)
                            boolean_encoded.append(encoded_single.reshape(-1, 1))
                        encoded_features[category] = np.concatenate(boolean_encoded, axis=1)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    elif category in ['categorical_high', 'text']:
                        # 处理所有文本特征 - 需要特殊处理
                        if hasattr(self.encoders[category], 'encode'):
                            # BERT编码器 - 处理所有文本特征
                            all_text_embeddings = []
                            for i in range(category_data.shape[1]):
                                text_data = category_data.iloc[:, i].astype(str)
                                text_list = text_data.tolist()
                                batch_size = 512 if torch.cuda.is_available() else 32 # 4096 512
                                text_embeddings = self.encoders[category].encode(text_list, batch_size=batch_size)
                                all_text_embeddings.append(text_embeddings)
                            # 将所有文本特征的嵌入拼接
                            encoded_features[category] = np.concatenate(all_text_embeddings, axis=1)
                        else:
                            # TF-IDF - 处理所有文本特征
                            all_text_features = []
                            for i in range(category_data.shape[1]):
                                text_data = category_data.iloc[:, i].fillna('无').astype(str)
                                all_text_features.append(text_data)
                            # 将所有文本特征合并
                            combined_text = pd.concat(all_text_features, axis=1)
                            encoded_features[category] = self.encoders[category].fit_transform(combined_text)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    else:
                        continue
                    
                    if pbar:
                        pbar.set_postfix({"类别": category, "特征数": len(available_features)})
                        pbar.update(1)
                except Exception as e:
                    print(f"  {category} 特征编码失败: {e}")
                    if pbar:
                        pbar.update(1)
                    continue
        
        finally:
            if pbar:
                pbar.close()
        
        # 打印总维度信息
        if self._transform_info_printed:
            total_dim = sum(encoded_features[cat].shape[1] for cat in encoded_features)
            print(f"  - 总特征维度: {total_dim}")
            self._transform_info_printed = False
        
        return encoded_features
    
    def fit_transform(self, df):
        """训练并编码特征"""
        self.fit(df)
        return self.transform(df)
    
    def transform_grouped(self, df, feature_weights, show_progress=False):
        """
        针对使用 BERT 编码的高维特征（text / categorical_high）
        按“相同权重”的特征先做内容拼接，再送入 BERT 编码，从而降低整体向量维度。
        
        其它低维特征（numerical / ordinal / categorical_low / boolean）保持原有逐特征编码方式不变。
        
        Args:
            df: pandas DataFrame，待编码的数据
            feature_weights: dict，特征名 -> 权重（来自 rag_weights 中的配置）
            show_progress: 是否显示进度条
        
        Returns:
            dict: {category: np.ndarray}，其中
                  - 非 BERT 类别与原 transform 输出保持一致
                  - BERT 类别在维度上改为“按权重分组后每组 1 个 768 维向量 * 组数”
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未fit，请先调用fit方法")
        
        # 将 list/array 单元格转为标量
        df = self._ensure_scalar_cells(df)
        
        encoded_features = {}
        
        # 使用active_feature_categories
        active_categories = self.active_feature_categories
        
        categories = [cat for cat in active_categories.keys() 
                      if any(f in df.columns for f in active_categories[cat])]
        
        # 创建进度条（仅在需要时显示）
        if show_progress:
            pbar = tqdm(total=len(categories), desc="编码特征（按权重分组）", unit="类别")
        else:
            pbar = None
        
        # ========= 先统一处理所有使用 BERT 的高维特征（text + categorical_high） =========
        bert_encoder = None
        if 'text' in self.encoders and hasattr(self.encoders['text'], 'encode'):
            bert_encoder = self.encoders['text']
        elif 'categorical_high' in self.encoders and hasattr(self.encoders['categorical_high'], 'encode'):
            bert_encoder = self.encoders['categorical_high']
        
        grouped_list = None
        if getattr(self, 'bert_group_info', None):
            grouped_list = self.bert_group_info
        else:
            bert_feature_names = []
            for cat in ['text', 'categorical_high']:
                if cat in active_categories:
                    for f in active_categories[cat]:
                        if f in df.columns:
                            bert_feature_names.append(f)
            if bert_encoder is not None and bert_feature_names:
                groups = {}
                for feat_name in bert_feature_names:
                    w = feature_weights.get(feat_name, 0.05)
                    key = float(w)
                    if key not in groups:
                        groups[key] = {'weight': float(w), 'features': []}
                    groups[key]['features'].append(feat_name)
                grouped_list = [groups[k] for k in sorted(groups.keys())]
                self.bert_group_info = grouped_list
        
        if bert_encoder is not None and grouped_list:
            all_group_embeddings = []
            for group in grouped_list:
                group_feats = group['features']
                def _join_row(row):
                    vals = []
                    for fn in group_feats:
                        v = row.get(fn, None)
                        if v is None:
                            text = ''
                        else:
                            text = str(v)
                            if text.strip() in ['无', '', 'nan', 'None']:
                                text = ''
                        vals.append(text)
                    joined = "\n".join(vals).strip()
                    return joined if joined else '[PAD]'
                combined_series = df.apply(_join_row, axis=1)
                text_list = combined_series.tolist()
                batch_size = 256 if torch.cuda.is_available() else 32
                group_emb = bert_encoder.encode(text_list, batch_size=batch_size)
                all_group_embeddings.append(group_emb)
            encoded_features['bert_grouped'] = np.concatenate(all_group_embeddings, axis=1)
        
        try:
            for category, features in active_categories.items():
                # 检查特征是否存在
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                    
                category_data = df[available_features]
                
                try:
                    if category == 'numerical':
                        # 处理所有数值特征
                        numeric_data = pd.to_numeric(category_data.iloc[:, 0], errors='coerce').fillna(0)
                        for i in range(1, category_data.shape[1]):
                            additional_data = pd.to_numeric(category_data.iloc[:, i], errors='coerce').fillna(0)
                            numeric_data = np.column_stack([numeric_data, additional_data])
                        encoded_features[category] = self.encoders[category].transform(numeric_data)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    
                    elif category == 'ordinal':
                        # 处理所有序数特征
                        filled_data = category_data.fillna('无')
                        encoded_features[category] = self.encoders[category].transform(filled_data)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    
                    elif category == 'categorical_low':
                        # 处理所有低基数分类特征
                        filled_data = category_data.fillna('无')
                        encoded_features[category] = self.encoders[category].transform(filled_data)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    
                    elif category == 'boolean':
                        # boolean：保持原有“逐特征编码”逻辑
                        boolean_encoded = []
                        for i in range(category_data.shape[1]):
                            if self.dataset_name in ('Multimodal_SDP', 'INSPIRE', 'MOVER_EPIC', 'MOVER_SIS'):
                                single_boolean = self._normalize_boolean_values(category_data.iloc[:, i])
                            else:
                                single_boolean = category_data.iloc[:, i].fillna('否').astype(str)
                            
                            encoded_single = self.encoders[category].transform(single_boolean)
                            boolean_encoded.append(encoded_single.reshape(-1, 1))
                        encoded_features[category] = np.concatenate(boolean_encoded, axis=1)
                        
                        if self._transform_info_printed:
                            actual_dim = encoded_features[category].shape[1]
                            feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                            print(f"  - {category}: {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    
                    elif category in ['categorical_high', 'text']:
                        # 对于 BERT 高维特征：
                        # - 如果已经通过 bert_encoder 统一处理成 'bert_grouped'，这里直接跳过
                        # - 如果 BERT 不可用（例如退回 TF-IDF），则保持原 per-category 逻辑
                        encoder = self.encoders.get(category, None)
                        if bert_encoder is not None:
                            # 已统一编码为 'bert_grouped'，此处不再单独处理
                            pass
                        else:
                            # BERT 不可用时的后备方案（TF-IDF），与原 transform 行为保持一致
                            if not hasattr(encoder, 'fit_transform'):
                                continue
                            all_text_features = []
                            for i in range(category_data.shape[1]):
                                text_data = category_data.iloc[:, i].fillna('无').astype(str)
                                all_text_features.append(text_data)
                            combined_text = pd.concat(all_text_features, axis=1)
                            encoded_features[category] = encoder.fit_transform(combined_text)
                            
                            if self._transform_info_printed:
                                actual_dim = encoded_features[category].shape[1]
                                feature_dim = actual_dim // len(available_features) if len(available_features) > 0 else 0
                                print(f"  - {category}(TF-IDF): {len(available_features)}个特征, 每特征{feature_dim}维, 类别总{actual_dim}维")
                    
                    else:
                        continue
                    
                    if pbar:
                        pbar.set_postfix({"类别": category, "特征数": len(available_features)})
                        pbar.update(1)
                except Exception as e:
                    print(f"  {category} 特征编码失败(分组): {e}")
                    if pbar:
                        pbar.update(1)
                    continue
        
        finally:
            if pbar:
                pbar.close()
        
        # 打印总维度信息
        if self._transform_info_printed:
            total_dim = sum(encoded_features[cat].shape[1] for cat in encoded_features)
            print(f"  - 总特征维度(分组BERT): {total_dim}")
            self._transform_info_printed = False
        
        return encoded_features
    
    def save_encoder(self, filepath):
        """保存编码器"""
        encoder_data = {
            'encoders': {},
            'feature_categories': self.active_feature_categories,
            'is_fitted': self.is_fitted,
            'dataset_name': self.dataset_name,
            'use_grouped_encoding': self.use_grouped_encoding,
            'bert_group_info': getattr(self, 'bert_group_info', []),
        }
        
        # 只保存可以序列化的编码器
        for category, encoder in self.encoders.items():
            if category in ['numerical', 'ordinal', 'categorical_low', 'boolean']:
                encoder_data['encoders'][category] = encoder
            elif category in ['categorical_high', 'text']:
                # 对于BERT编码器，只保存标记
                encoder_data['encoders'][category] = 'bert_encoder'
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
    
    def load_encoder(self, filepath):
        """加载编码器"""
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        # 加载数据集名称（如果存在）
        self.dataset_name = encoder_data.get('dataset_name', '')
        
        # 加载编码方式设置（向后兼容：如果不存在则默认为False，使用独立编码）
        self.use_grouped_encoding = encoder_data.get('use_grouped_encoding', False)
        
        # 根据数据集类型选择特征配置
        if self.dataset_name == 'Multimodal_SDP':
            self.active_feature_categories = self.feature_categories_MMSDP
        elif self.dataset_name == 'INSPIRE':
            self.active_feature_categories = self.feature_categories_INSPIRE
        elif self.dataset_name == 'MOVER_EPIC':
            self.active_feature_categories = self.feature_categories_MOVER_EPIC
        elif self.dataset_name == 'MOVER_SIS':
            self.active_feature_categories = self.feature_categories_MOVER_SIS
        else:
            self.active_feature_categories = self.feature_categories
        
        # 加载保存的特征配置（用于兼容性检查）
        saved_feature_categories = encoder_data.get('feature_categories', {})
        if saved_feature_categories:
            # 如果保存的配置与当前配置不一致，使用保存的配置
            self.active_feature_categories = saved_feature_categories
        
        self.feature_categories = self.active_feature_categories
        self.is_fitted = encoder_data['is_fitted']
        self.bert_group_info = encoder_data.get('bert_group_info', [])
        
        # 重新初始化编码器
        self.encoders = self._init_encoders()
        
        # 加载可序列化的编码器
        for category, encoder in encoder_data['encoders'].items():
            if category in ['numerical', 'ordinal', 'categorical_low', 'boolean']:
                self.encoders[category] = encoder
            elif category in ['categorical_high', 'text'] and encoder == 'bert_encoder':
                # BERT编码器已经在_init_encoders中初始化
                pass
