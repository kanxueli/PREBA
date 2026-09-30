"""
文本编码器模块
支持多种BERT模型对文本化案例进行编码
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertModel


# 预定义模型配置
MODEL_CONFIGS = {
    'bert-base-chinese': {
        'path': '/home/data2/LLM_benchmarking/bert-base-chinese',
        'max_length': 512,
        'fallback': 'bert-base-chinese'  # 如果本地路径不存在，使用HuggingFace名称
    },
    'clinical-modernbert': {
        'path': '/home/data2/LLM_benchmarking/Clinical_ModernBERT',
        'max_length': 512
    },
    'medbert': {
        'path': '/home/data2/LLM_benchmarking/medbert-base-chinese',
        'max_length': 512
    },
    'chinese-modernbert': {
        'path': '/home/data2/LLM_benchmarking/ChineseModernBert',
        'max_length': 512
    },
    'bert-base-uncased': {
        'path': '/home/data2/LLM_benchmarking/bert-base-uncased',
        'max_length': 512,
        'fallback': 'bert-base-uncased'
    },
    'medbert-base': {
        'path': '/home/data2/LLM_benchmarking/medBERT-base',
        'max_length': 512
    },
    'modernbert-base': {
        'path': '/home/data2/LLM_benchmarking/ModernBERT-base',
        'max_length': 512
    }
}


class TextEncoder:
    """
    统一的文本编码器
    支持通过模型名称或路径指定不同的BERT模型
    """
    
    def __init__(self, model_name=None, model_path=None, device=None, max_length=None):
        """
        初始化文本编码器
        
        Args:
            model_name: 模型名称，可选值: 'bert-base-chinese', 'clinical-modernbert', 'medbert', 'chinese-modernbert'
                       如果指定了model_name，会使用预定义的配置
            model_path: 模型路径，如果指定了model_path，会直接使用该路径
                       如果同时指定了model_name和model_path，model_path优先
            device: 设备，如果为None则自动检测
            max_length: 最大文本长度，如果为None则使用模型配置中的默认值
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确定模型路径和最大长度
        if model_path is not None:
            # 直接使用指定的路径
            final_model_path = model_path
            if max_length is None:
                max_length = 512  # 默认值
        elif model_name is not None and model_name.lower() in MODEL_CONFIGS:
            # 使用预定义的模型配置
            config = MODEL_CONFIGS[model_name.lower()]
            final_model_path = config['path']
            if max_length is None:
                max_length = config.get('max_length', 512)
            
            # 检查路径是否存在，如果不存在且有fallback，使用fallback
            if not os.path.exists(final_model_path) and 'fallback' in config:
                final_model_path = config['fallback']
        else:
            # 默认使用bert-base-chinese
            config = MODEL_CONFIGS['bert-base-chinese']
            final_model_path = config['path']
            if not os.path.exists(final_model_path):
                final_model_path = config.get('fallback', 'bert-base-chinese')
            if max_length is None:
                max_length = config.get('max_length', 512)
        
        self.max_length = max_length
        self.model_path = final_model_path
        
        # 加载模型
        print(f"加载模型: {final_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(final_model_path)
        # Clinical_ModernBERT 检查点中不包含 pooler 权重：
        # - 当使用 clinical-modernbert 预设名称，或路径中包含 Clinical_ModernBERT 时，
        #   统一用 BertModel 且关闭 pooling 层，避免未初始化 pooler 权重的警告。
        name_lower = (model_name or "").lower()
        is_clinical_modernbert = (
            name_lower == "clinical-modernbert"
            or "clinical_modernbert" in final_model_path.lower()
            or "clinical-modernbert" in final_model_path.lower()
        )
        if is_clinical_modernbert:
            self.model = BertModel.from_pretrained(final_model_path, add_pooling_layer=False)
        else:
            self.model = AutoModel.from_pretrained(final_model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"文本编码器初始化完成，设备: {self.device}, 嵌入维度: {self.embedding_dim}, 最大长度: {self.max_length}")
    
    def encode(self, texts, batch_size=2048, return_numpy=True):
        """
        编码文本为向量
        
        Args:
            texts: 文本字符串或文本列表
            batch_size: 批处理大小
            return_numpy: 是否返回numpy数组
        
        Returns:
            numpy array 或 torch tensor: 编码后的向量，shape为 [n_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) == 0:
            if return_numpy:
                return np.zeros((0, self.embedding_dim))
            else:
                return torch.zeros((0, self.embedding_dim), device=self.device)
        
        all_embeddings = []
        
        # 批处理编码
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 处理空文本
                processed_texts = []
                for text in batch_texts:
                    if pd.isna(text) or str(text).strip() in ['无', '', 'nan', 'None', '']:
                        processed_texts.append('[PAD]')
                    else:
                        processed_texts.append(str(text))
                
                # Tokenization
                inputs = self.tokenizer(
                    processed_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 使用[CLS] token的表示作为句子向量
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
                
                all_embeddings.append(batch_embeddings.cpu())
                
                # 清理GPU缓存
                del inputs, outputs, batch_embeddings
        
        # 合并所有批次
        if return_numpy:
            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings
    
    def encode_query_case(self, text):
        """
        编码查询案例文本
        
        Args:
            text: 查询案例的文本表示
        
        Returns:
            numpy array: 编码后的向量，shape为 [embedding_dim]
        """
        embedding = self.encode([text], return_numpy=True)
        return embedding[0]  # 返回单个向量
    
    def encode_retrieved_cases(self, texts, return_numpy=True):
        """
        编码检索案例文本列表
        
        Args:
            texts: 检索案例的文本表示列表
            return_numpy: 是否返回numpy数组
        
        Returns:
            numpy array 或 torch tensor: 编码后的向量，shape为 [n_cases, embedding_dim]
        """
        if len(texts) == 0:
            if return_numpy:
                return np.zeros((0, self.embedding_dim))
            else:
                return torch.zeros((0, self.embedding_dim), device=self.device)
        
        return self.encode(texts, return_numpy=return_numpy)


# 向后兼容的别名类
class ClinicalModernBERTEncoder(TextEncoder):
    """
    临床ModernBERT编码器（向后兼容）
    使用Clinical_ModernBERT模型对文本进行编码
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        """
        初始化Clinical_ModernBERT编码器（简单包装 TextEncoder）
        """
        if model_path is None:
            super().__init__(model_name='clinical-modernbert', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)


class MedBERTEncoder(TextEncoder):
    """
    MedBERT编码器（向后兼容）
    使用medbert-base-chinese模型对文本进行编码
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        """
        初始化MedBERT编码器
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
            device: 设备，如果为None则自动检测
            max_length: 最大文本长度，默认512
        """
        if model_path is None:
            super().__init__(model_name='medbert', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)


class ChineseModernBERTEncoder(TextEncoder):
    """
    ChineseModernBERT编码器（向后兼容）
    使用ChineseModernBert模型对文本进行编码
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        """
        初始化ChineseModernBERT编码器
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
            device: 设备，如果为None则自动检测
            max_length: 最大文本长度，默认512
        """
        if model_path is None:
            super().__init__(model_name='chinese-modernbert', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)


class BertBaseUncasedEncoder(TextEncoder):
    """
    BERT-base-uncased 编码器
    使用 bert-base-uncased 对文本进行编码（英文，适用于 INSPIRE 等英文数据集）
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        if model_path is None:
            super().__init__(model_name='bert-base-uncased', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)


class MedBERTBaseEncoder(TextEncoder):
    """
    MedBERT-base 编码器
    使用 medBERT-base 对文本进行编码（英文医学 BERT）
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        if model_path is None:
            super().__init__(model_name='medbert-base', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)


class ModernBERTBaseEncoder(TextEncoder):
    """
    ModernBERT-base 编码器
    使用 ModernBERT-base 对文本进行编码
    """
    
    def __init__(self, model_path=None, device=None, max_length=512):
        if model_path is None:
            super().__init__(model_name='modernbert-base', device=device, max_length=max_length)
        else:
            super().__init__(model_path=model_path, device=device, max_length=max_length)
