import faiss
import numpy as np
import pandas as pd
import pickle
import os
import torch
from .rag_encoder import HeterogeneousFeatureEncoder

class RAGRetriever:
    """
    RAG检索器
    负责在FAISS向量数据库中检索最相关的案例
    """
    
    def __init__(self, index_path='faiss_index.faiss', mapping_path='faiss_mappings.pkl', encoder_path='rag_encoder.pkl', dataset_name='', ran_path=None):
        """
        初始化RAG检索器
        
        Args:
            index_path: FAISS索引文件路径
            mapping_path: 映射关系文件路径
            encoder_path: 编码器文件路径
            dataset_name: 数据集名称，用于选择特征配置和权重
            ran_path: RAN模型路径（可选），如果提供则加载并使用RAN进行查询向量优化
        """
        self.index = None
        self.id_mapping = {}
        self.feature_weights = {}
        self.encoder = None
        self.is_loaded = False
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.encoder_path = encoder_path
        self.dataset_name = dataset_name
        self.ran_model = None
        self.ran_device = None
        
        # 加载数据库
        self.load_database(index_path, mapping_path, encoder_path)
        
        # 加载RAN模型（如果提供路径）
        if ran_path is not None and os.path.exists(ran_path):
            self.load_ran_model(ran_path)
        elif ran_path is not None:
            print(f"⚠ RAN模型路径不存在: {ran_path}，将不使用RAN")
    
    def load_database(self, index_path, mapping_path, encoder_path=None):
        """加载FAISS数据库"""
        try:
            # 检查文件是否存在
            if not os.path.exists(index_path) or not os.path.exists(mapping_path):
                print(f"数据库文件不存在: {index_path} 或 {mapping_path}")
                return False
            
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 加载映射关系
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                self.id_mapping = mapping_data['id_mapping']
                self.feature_weights = mapping_data['feature_weights']
            
            # 加载编码器
            if encoder_path and os.path.exists(encoder_path):
                # 从映射数据中获取数据集名称（如果存在）
                saved_dataset_name = mapping_data.get('dataset_name', self.dataset_name)
                # 从编码器中获取use_grouped_encoding设置（先加载编码器才能获取）
                temp_encoder = HeterogeneousFeatureEncoder(dataset_name=saved_dataset_name)
                temp_encoder.load_encoder(encoder_path)
                saved_use_grouped = getattr(temp_encoder, 'use_grouped_encoding', False)
                
                # 使用保存的设置初始化编码器
                self.encoder = HeterogeneousFeatureEncoder(
                    dataset_name=saved_dataset_name,
                    use_grouped_encoding=saved_use_grouped
                )
                self.encoder.load_encoder(encoder_path)
                # 更新dataset_name以保持一致性
                self.dataset_name = saved_dataset_name
                print(f"✓ 编码器加载成功: {encoder_path} (编码方式: {'分组编码' if saved_use_grouped else '独立编码'})")
            else:
                print(f"⚠ 编码器文件不存在: {encoder_path}")
                return False
            
            self.is_loaded = True
            # 打印数据库详细信息
            print(f"\n数据库详细信息:")
            print(f"  - 总记录数: {len(self.id_mapping)}")
            print(f"  - 向量维度: {self.index.d}")
            print(f"  - 索引类型: {type(self.index).__name__}")
            print(f"  - 索引路径: {self.index_path}")

            return True
            
        except Exception as e:
            print(f"RAG检索器加载失败: {e}")
            return False
    
    def retrieve_similar_cases(self, query_row, k=5):
        """
        检索最相关的Top-K案例（按相似度排序）
        
        Args:
            query_row: 查询样本 (pandas Series)
            k: 检索数量
        
        Returns:
            list: 检索结果列表，每个元素包含相似度和原始索引
        """
        if not self.is_loaded:
            print("检索器未加载，无法进行检索")
            return []
        
        try:
            query_df = pd.DataFrame([query_row])
            # 使用与构建数据库时一致的编码方案
            if self.encoder.use_grouped_encoding:
                # 使用分组编码
                encoded_features = self.encoder.transform_grouped(
                    query_df,
                    feature_weights=self._get_features_default_weights(),
                    show_progress=False
                )
            else:
                # 使用独立编码
                encoded_features = self.encoder.transform(
                    query_df,
                    show_progress=False
                )
            
            # 构建查询向量
            query_vector = self._build_query_vector(encoded_features, query_row)
            
            query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
            
            # 使用 RAN 对查询向量做适配
            if self.ran_model is not None:
                with torch.no_grad():
                    q_tensor = torch.from_numpy(query_vector).to(self.ran_device).unsqueeze(0)  # [1, D]
                    q_adapt = self.ran_model.forward_query(q_tensor).squeeze(0).cpu().numpy()
                query_vector = np.ascontiguousarray(q_adapt, dtype=np.float32)
            
            # 归一化
            faiss.normalize_L2(query_vector.reshape(1, -1))
            query_vector = query_vector.flatten()
            
            # FAISS检索，检索Top-K
            query_vector = query_vector.reshape(1, -1).astype('float32')
            similarities, indices = self.index.search(query_vector, k)
            
            # 构建结果列表
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx in self.id_mapping:
                    original_idx = self.id_mapping[idx]
                    results.append({
                        'similarity': float(sim),
                        'original_index': original_idx,
                        'faiss_index': int(idx)
                    })
            
            # 按相似度排序并返回
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
            
        except Exception as e:
            import traceback
            error_msg = str(e) if e else "未知错误"
            traceback_str = traceback.format_exc()
            print(f"检索过程出错: {error_msg}")
            print(f"详细错误信息:\n{traceback_str}")
            return []
    
    def _build_query_vector(self, encoded_features, query_row):
        """构建查询向量"""
        vector_parts = []
        
        # 使用active_feature_categories确保使用正确的特征配置
        feature_categories = getattr(self.encoder, 'active_feature_categories', self.encoder.feature_categories)
        
        # 如果使用分组编码，需要特殊处理 bert_grouped
        if self.encoder.use_grouped_encoding and 'bert_grouped' in encoded_features and getattr(self.encoder, 'bert_group_info', None):
            # 先处理所有非BERT类别
            for category, features in feature_categories.items():
                if category in ['text', 'categorical_high']:
                    continue
                if category not in encoded_features:
                    continue
                try:
                    category_data = encoded_features[category]
                    
                    if len(category_data) == 0:
                        print(f"警告: 类别 {category} 数据为空")
                        continue
                    
                    if category_data.ndim == 1:
                        category_vector = category_data
                    else:
                        category_vector = category_data[0]
                    
                    if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                        category_vector = category_vector.flatten()
                    
                    weighted_vectors = []
                    for i, feature_name in enumerate(features):
                        if feature_name in query_row.index:
                            feature_weights = self._get_features_default_weights()
                            feature_weight = feature_weights.get(feature_name, 0.05)
                            
                            feature_dim = len(category_vector) // len(features)
                            start_idx = i * feature_dim
                            end_idx = (i + 1) * feature_dim
                            
                            if end_idx <= len(category_vector):
                                feature_vector = category_vector[start_idx:end_idx]
                                category_weight = self._get_category_weight(category, len(feature_vector))
                                weighted_feature = feature_vector * feature_weight * category_weight
                                weighted_vectors.append(weighted_feature)
                    
                    if weighted_vectors:
                        vector_parts.append(np.concatenate(weighted_vectors))
                
                except (IndexError, KeyError) as e:
                    print(f"  警告: 类别 {category} 编码失败: {e}")
                    continue
            
            # 处理 bert_grouped
            try:
                category_data = encoded_features['bert_grouped']
                if len(category_data) == 0:
                    print("警告: BERT 分组特征数据为空")
                else:
                    if category_data.ndim == 1:
                        category_vector = category_data
                    else:
                        category_vector = category_data[0]
                    
                    if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                        category_vector = category_vector.flatten()
                    
                    groups = self.encoder.bert_group_info
                    group_vectors = []
                    feature_dim = 768
                    offset = 0
                    for group in groups:
                        end = offset + feature_dim
                        if end > len(category_vector):
                            break
                        group_vec = category_vector[offset:end]
                        offset = end
                        
                        group_weight = float(group.get('weight', 0.05))
                        category_weight = self._get_category_weight('text', len(group_vec))
                        weighted_group = group_vec * group_weight * category_weight
                        group_vectors.append(weighted_group)
                    
                    if group_vectors:
                        vector_parts.append(np.concatenate(group_vectors))
            except (IndexError, KeyError) as e:
                print(f"  警告: BERT 分组特征编码失败: {e}")
        else:
            # 使用独立编码：统一处理所有类别
            for category, features in feature_categories.items():
                if category not in encoded_features:
                    continue
                try:
                    category_data = encoded_features[category]
                    
                    if len(category_data) == 0:
                        print(f"警告: 类别 {category} 数据为空")
                        continue
                    
                    if category_data.ndim == 1:
                        category_vector = category_data
                    else:
                        category_vector = category_data[0]
                    
                    if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                        category_vector = category_vector.flatten()
                    
                    # 对每个特征分别应用权重（与旧版本逻辑完全一致）
                    weighted_vectors = []
                    for i, feature_name in enumerate(features):
                        if feature_name in query_row.index:
                            # 获取特征权重
                            feature_weights = self._get_features_default_weights()
                            feature_weight = feature_weights.get(feature_name, 0.05)
                            
                            # 计算该特征在向量中的位置
                            if category in ['categorical_high', 'text']:
                                # BERT特征：每个768维
                                start_idx = i * 768
                                end_idx = (i + 1) * 768
                            else:
                                # 其他特征：按实际维度
                                feature_dim = len(category_vector) // len(features)
                                start_idx = i * feature_dim
                                end_idx = (i + 1) * feature_dim
                            
                            # 对该特征部分应用权重
                            if end_idx <= len(category_vector):
                                feature_vector = category_vector[start_idx:end_idx]
                                
                                # 获取类别权重
                                category_weight = self._get_category_weight(category, len(feature_vector))
                                
                                # 应用混合权重
                                weighted_feature = feature_vector * feature_weight * category_weight
                                weighted_vectors.append(weighted_feature)
                    
                    if weighted_vectors:
                        vector_parts.append(np.concatenate(weighted_vectors))
                
                except (IndexError, KeyError) as e:
                    print(f"  警告: 类别 {category} 编码失败: {e}")
                    continue
        
        if not vector_parts:
            print("  错误: 所有特征编码都失败了")
            raise ValueError("所有特征编码都失败了")
        
        full_vector = np.concatenate(vector_parts)
        # print(f"  调试: 查询向量总长度: {len(full_vector)}")
        return full_vector
    
    def _get_features_default_weights(self):
        """获取所有已选特征的默认权重配置 - 与数据库保持一致"""
        from .rag_weights import (
            get_feature_weights,
            get_feature_weights_MMSDP,
            get_feature_weights_INSPIRE,
            get_feature_weights_MOVER_EPIC,
            get_feature_weights_MOVER_SIS,
        )
        if self.dataset_name == 'Multimodal_SDP':
            return get_feature_weights_MMSDP()
        if self.dataset_name == 'INSPIRE':
            return get_feature_weights_INSPIRE()
        if self.dataset_name == 'MOVER_EPIC':
            return get_feature_weights_MOVER_EPIC()
        if self.dataset_name == 'MOVER_SIS':
            return get_feature_weights_MOVER_SIS()
        return get_feature_weights()
    
    def _get_category_weight(self, category, encoded_vector_length):
        """根据实际编码长度动态计算类别权重"""
        from .rag_weights import calculate_category_weight
        return calculate_category_weight(category, encoded_vector_length)
    
    def load_ran_model(self, ran_path):
        """
        加载RAN模型
        
        Args:
            ran_path: RAN模型文件路径
        """
        try:
            import sys
            import os
            # 添加项目根目录到路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from models.RAN4RAG import load_ran
            
            # 确定设备
            self.ran_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 加载模型
            self.ran_model = load_ran(ran_path, map_location=str(self.ran_device))
            self.ran_model = self.ran_model.to(self.ran_device)
            self.ran_model.eval()
            
            print(f"✓ RAN模型加载成功: {ran_path}")
            print(f"  - 设备: {self.ran_device}")
            print(f"  - 输入维度: {self.ran_model.input_dim}")
        except Exception as e:
            print(f"⚠ RAN模型加载失败: {e}，将不使用RAN")
            import traceback
            traceback.print_exc()
            self.ran_model = None
            self.ran_device = None
    
    def get_retrieval_stats(self, retrieval_results):
        """获取检索统计信息"""
        if not retrieval_results:
            return {
                'total_retrieved': 0,
                'avg_similarity': 0.0,
                'similarity_range': {'min': 0.0, 'max': 0.0}
            }
        
        similarities = [r['similarity'] for r in retrieval_results]
        return {
            'total_retrieved': len(retrieval_results),
            'avg_similarity': np.mean(similarities),
            'similarity_range': {
                'min': min(similarities),
                'max': max(similarities)
            }
        }

class RAGDataMapper:
    """
    RAG数据映射器
    负责将检索结果映射回原始数据格式
    """
    
    def __init__(self, original_df):
        """
        初始化数据映射器
        
        Args:
            original_df: 原始数据DataFrame
        """
        self.original_df = original_df
    
    def map_to_original_data(self, retrieval_results):
        """
        将检索结果映射回原始数据格式
        
        Args:
            retrieval_results: 检索结果列表
        
        Returns:
            pd.DataFrame: 原始格式的相关案例数据
        """
        if not retrieval_results:
            return pd.DataFrame()
        
        original_cases = []
        
        for i, result in enumerate(retrieval_results):
            original_idx = result['original_index']
            
            # 获取原始数据
            original_case = self.original_df.iloc[original_idx].copy()
            
            # 添加检索信息
            original_case['retrieval_similarity'] = result['similarity']
            original_case['retrieval_rank'] = i + 1
            original_case['retrieval_faiss_index'] = result['faiss_index']
            
            original_cases.append(original_case)
        
        return pd.DataFrame(original_cases)
    
    def get_retrieval_summary(self, retrieval_results):
        """获取检索结果摘要"""
        if not retrieval_results:
            return "无检索结果"
        
        stats = {
            'total_retrieved': len(retrieval_results),
            'avg_similarity': np.mean([r['similarity'] for r in retrieval_results]),
            'similarity_range': {
                'min': min([r['similarity'] for r in retrieval_results]),
                'max': max([r['similarity'] for r in retrieval_results])
            }
        }
        
        summary = f"检索到 {stats['total_retrieved']} 个相关案例，"
        summary += f"平均相似度: {stats['avg_similarity']:.3f}，"
        summary += f"相似度范围: {stats['similarity_range']['min']:.3f} - {stats['similarity_range']['max']:.3f}"
        
        return summary
