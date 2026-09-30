import faiss
import numpy as np
import pandas as pd
import pickle
import os
import time
from tqdm import tqdm
from .rag_encoder import HeterogeneousFeatureEncoder

class FAISSVectorDatabase:
    """
    FAISS向量数据库构建器
    负责将异构特征编码为向量并构建FAISS索引
    """
    
    def __init__(self, feature_weights=None, database_path=None, dataset_name='', use_grouped_encoding=False):
        """
        初始化FAISS向量数据库
        
        Args:
            feature_weights: 特征权重配置字典
            database_path: 数据库存储路径
            dataset_name: 数据集名称，用于选择特征配置和权重
            use_grouped_encoding: 是否使用分组编码（True=按权重分组编码BERT特征，False=独立编码每个特征）
        """
        self.dataset_name = dataset_name
        self.use_grouped_encoding = use_grouped_encoding
        self.feature_weights = feature_weights or self._get_features_default_weights()
        self.encoder = HeterogeneousFeatureEncoder(dataset_name=dataset_name, use_grouped_encoding=use_grouped_encoding)
        self.index = None
        self.id_mapping = {}  # 向量ID到原始数据索引的映射
        self.feature_dimensions = {}
        self.database_path = database_path or "faiss_index.faiss"
        self.mapping_path = database_path.replace('.faiss', '_mappings.pkl') if database_path else "faiss_mappings.pkl"
        self.encoder_path = database_path.replace('.faiss', '_encoder.pkl') if database_path else "rag_encoder.pkl"
        # 当前编码版本（用于区分编码方式：1=原始独立编码，2=分组编码，3=支持两种方式）
        # 版本3支持根据use_grouped_encoding参数选择编码方式
        self.encoding_version = 3
    
    def _get_features_default_weights(self):
        """获取所有已选特征的默认权重配置 - 基于医学重要性"""
        # 从共享模块导入权重配置
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
    
    def _save_encoded_features(self, encoded_features, filepath, original_length):
        """保存编码后的特征到文件"""
        import pickle
        try:
            cache_data = {
                'encoded_features': encoded_features,
                'original_length': original_length,
                'feature_categories': self.encoder.feature_categories,
                'timestamp': time.time(),
                'encoding_version': self.encoding_version,
                'use_grouped_encoding': self.use_grouped_encoding,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"    编码特征已保存，原始长度: {original_length}")
        except Exception as e:
            print(f"    保存编码特征失败: {e}")
    
    def _load_encoded_features(self, filepath, df):
        """加载已编码的特征"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 编码版本不一致则强制重新编码
            # 版本3兼容版本1和2，但需要检查use_grouped_encoding是否一致
            cached_version = cache_data.get('encoding_version', 1)
            cached_use_grouped = cache_data.get('use_grouped_encoding', False)
            
            if cached_version != self.encoding_version:
                # 版本号不同，需要重新编码
                print(f"    编码版本变化 (缓存版本: {cached_version}, 当前版本: {self.encoding_version})，需要重新编码")
                return None
            elif cached_version == 3 and cached_use_grouped != self.use_grouped_encoding:
                # 版本3但编码方式不同，需要重新编码
                print(f"    编码方式变化 (缓存: {'分组编码' if cached_use_grouped else '独立编码'}, 当前: {'分组编码' if self.use_grouped_encoding else '独立编码'})，需要重新编码")
                return None
            
            # 检查数据集长度是否一致
            if cache_data['original_length'] != len(df):
                print(f"    数据集长度不匹配 (缓存: {cache_data['original_length']}, 当前: {len(df)})，需要重新编码")
                return None
            
            # 检查特征类别是否一致（使用active_feature_categories）
            saved_categories = cache_data.get('feature_categories', {})
            current_categories = self.encoder.active_feature_categories
            if saved_categories != current_categories:
                print("    特征类别配置不匹配，需要重新编码")
                return None
            
            print(f"成功加载编码特征缓存 (长度: {cache_data['original_length']})")
            return cache_data['encoded_features']
            
        except Exception as e:
            print(f"    加载编码特征缓存失败: {e}，需要重新编码")
            return None

    # index_type在此处定义包括 IVFFlat, Flat
    def build_database(self, df, index_type='Flat', force_rebuild=False, force_reencode=False, ran_path=None):
        """
        构建FAISS向量数据库
        
        Args:
            df: 训练数据DataFrame
            index_type: FAISS索引类型 ('IVFFlat' 或 'Flat') Flat是暴力搜索，IVFFlat是倒排索引加聚类
            force_rebuild: 是否强制重建数据库
            force_reencode: 是否强制重新编码特征（忽略缓存）
            ran_path: RAN模型路径（可选），如果是both-adaptor模式，会对database向量应用adapter
        
        Returns:
            bool: 是否成功构建数据库
        """
        # 检查数据库是否已存在
        if not force_rebuild and self._check_database_exists():
            print("FAISS数据库已存在，跳过构建过程")
            return True
        
        try:
            print("开始构建FAISS向量数据库...")
            print(f"数据集大小: {len(df)} 条记录")
            
            # 1. 训练编码器
            print("  Fitting encoder...")
            self.encoder.fit(df)
            
            # 2. 编码特征并构建向量,先尝试加载已编码的特征（除非强制重新编码）
            encoded_features_path = self.database_path.replace('.faiss', '_encoded_features.pkl')
            encoded_features = None
            if not force_reencode:
                encoded_features = self._load_encoded_features(encoded_features_path, df)
            
            if encoded_features is None:
                print("    开始编码特征...")
                # 根据use_grouped_encoding选择编码方式
                if self.use_grouped_encoding:
                    # 使用分组编码
                    encoded_features = self.encoder.transform_grouped(
                        df,
                        feature_weights=self._get_features_default_weights(),
                        show_progress=True
                    )
                else:
                    # 使用独立编码（原始方式）
                    encoded_features = self.encoder.transform(
                        df,
                        show_progress=True
                    )
                
                # 保存编码后的特征
                self._save_encoded_features(encoded_features, encoded_features_path, len(df))
                print(f"    已保存编码特征到: {encoded_features_path}")
            
            # 检查编码后数据集大小，确保与原始数据集一致
            if isinstance(encoded_features, dict) and len(encoded_features) > 0:
                first_category = list(encoded_features.keys())[0]
                encoded_sample_count = len(encoded_features[first_category])
                print(f"编码后数据集大小: {encoded_sample_count} 条记录")
                
                if encoded_sample_count != len(df):
                    print(f"警告：编码前后数据集大小不一致！原始: {len(df)}, 编码后: {encoded_sample_count}")
                    return False
            else:
                print("警告：编码结果结构异常或为空，将重新编码一次以修复旧缓存")
                # 强制重新编码一次（不再使用缓存）
                if self.use_grouped_encoding:
                    encoded_features = self.encoder.transform_grouped(
                        df,
                        feature_weights=self._get_features_default_weights(),
                        show_progress=True
                    )
                else:
                    encoded_features = self.encoder.transform(
                        df,
                        show_progress=True
                    )
                if not isinstance(encoded_features, dict) or len(encoded_features) == 0:
                    print("错误：重新编码后结果仍然为空，构建失败")
                    return False
            
            # 检查是否需要应用 RAN database adapter
            need_apply_adapter = False
            ran_model = None
            if ran_path is not None and os.path.exists(ran_path):
                try:
                    from models.RAN4RAG import load_ran
                    import torch
                    ran_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    ran_model = load_ran(ran_path, map_location=str(ran_device))
                    if ran_model.adapt_mode == "both-adaptor":
                        need_apply_adapter = True
                        print(f"\n检测到 both-adaptor 模式的 RAN 模型，将对 database 向量应用 adapter")
                except Exception as e:
                    print(f"⚠ 加载 RAN 模型失败: {e}，将跳过 database adapter 应用")
            
            # 构建加权特征向量（如果不需要应用 adapter，直接归一化；否则先不归一化）
            if need_apply_adapter:
                print("  构建未归一化的加权特征向量（将应用 RAN adapter）...")
                all_vectors_unnorm = self._build_weighted_vectors(encoded_features, df, normalize=False)
                
                # 应用 RAN database adapter
                print("  应用 RAN database adapter...")
                try:
                    import torch
                    import torch.nn.functional as F
                    ran_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    ran_model = ran_model.to(ran_device)
                    ran_model.eval()
                    
                    # 批量处理向量
                    batch_size = 5000
                    all_vectors_adapted = []
                    with torch.no_grad():
                        for i in tqdm(range(0, len(all_vectors_unnorm), batch_size), desc="  适应database向量", leave=False):
                            batch_vectors = all_vectors_unnorm[i:i+batch_size]
                            batch_tensor = torch.from_numpy(np.array(batch_vectors, dtype=np.float32)).to(ran_device)
                            
                            # 通过 database adapter
                            adapted_batch = ran_model.forward_database(batch_tensor)
                            
                            # 归一化
                            adapted_batch_norm = F.normalize(adapted_batch, p=2, dim=-1, eps=1e-30)
                            all_vectors_adapted.append(adapted_batch_norm.cpu().numpy())
                    
                    # 合并所有向量并转换为列表格式
                    all_vectors_array = np.vstack(all_vectors_adapted).astype(np.float32)
                    # 确保向量已归一化（使用FAISS归一化再次确认）
                    faiss.normalize_L2(all_vectors_array)
                    all_vectors = [all_vectors_array[i] for i in range(len(all_vectors_array))]
                    
                    print("  ✓ RAN database adapter 应用完成")
                except Exception as e:
                    print(f"  ⚠ 应用 RAN adapter 失败: {e}，将使用原始向量")
                    import traceback
                    traceback.print_exc()
                    # 如果失败，回退到正常流程：归一化原始向量
                    all_vectors = self._build_weighted_vectors(encoded_features, df, normalize=True)
            else:
                # 正常流程：直接构建归一化向量
                all_vectors = self._build_weighted_vectors(encoded_features, df, normalize=True)
            
            # 清理显存
            del encoded_features
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # 3. 创建FAISS索引
            print("  创建FAISS索引...")
            self._create_faiss_index(all_vectors, index_type)
            
            # 4. 保存映射关系
            print("  保存映射关系...")
            self._save_mappings(df)
            
            # 5. 保存编码器
            print("  保存特征编码器...")
            self.encoder.save_encoder(self.encoder_path)
            
            print(f"FAISS数据库构建完成，包含 {len(all_vectors)} 条记录")
            return True
            
        except Exception as e:
            print(f"FAISS数据库构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_database_exists(self):
        """检查数据库是否已存在"""
        required_files = [self.database_path, self.mapping_path, self.encoder_path]
        return all(os.path.exists(f) for f in required_files)
    
    def _build_weighted_vectors(self, encoded_features, df, normalize=True):
        """
        构建加权特征向量
        
        Args:
            encoded_features: 编码后的特征字典
            df: 原始数据DataFrame
            normalize: 是否对向量进行L2归一化（默认True，保持向后兼容）
        
        Returns:
            list: 向量列表（已归一化或未归一化，取决于normalize参数）
        """
        vectors = []
        
        # 创建进度条
        with tqdm(total=len(df), desc="构建特征向量", unit="样本") as pbar:
            for local_idx, (original_idx, row) in enumerate(df.iterrows()):
                vector_parts = []
                
                # 按特征类别构建向量
                # 使用active_feature_categories确保使用正确的特征配置
                feature_categories = getattr(self.encoder, 'active_feature_categories', self.encoder.feature_categories)
                
                # 如果使用分组编码，需要特殊处理 bert_grouped
                if self.use_grouped_encoding and 'bert_grouped' in encoded_features and getattr(self.encoder, 'bert_group_info', None):
                    # 先处理所有非BERT类别
                    for category, features in feature_categories.items():
                        if category in ['text', 'categorical_high']:
                            continue
                        if category not in encoded_features:
                            continue
                        
                        category_vector = encoded_features[category][local_idx]
                        if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                            category_vector = category_vector.flatten()
                        
                        weighted_vectors = []
                        for i, feature_name in enumerate(features):
                            if feature_name in df.columns:
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
                    
                    # 处理 bert_grouped
                    category_vector = encoded_features['bert_grouped'][local_idx]
                    if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                        category_vector = category_vector.flatten()
                    
                    groups = self.encoder.bert_group_info
                    group_vectors = []
                    feature_dim = 768  # BERT-base 维度
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
                else:
                    # 使用独立编码：统一处理所有类别
                    for category, features in feature_categories.items():
                        if category not in encoded_features:
                            continue
                        
                        category_vector = encoded_features[category][local_idx]
                        if hasattr(category_vector, 'ndim') and category_vector.ndim > 1:
                            category_vector = category_vector.flatten()
                        
                        # 对每个特征分别应用权重
                        weighted_vectors = []
                        for i, feature_name in enumerate(features):
                            if feature_name in df.columns:
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
                
                # 拼接所有特征向量
                if vector_parts:
                    try:
                        full_vector = np.concatenate(vector_parts)
                        
                        # 确保数组是连续的
                        full_vector = np.ascontiguousarray(full_vector, dtype=np.float32)
                        
                        # 根据参数决定是否归一化
                        if normalize:
                            faiss.normalize_L2(full_vector.reshape(1, -1))
                            full_vector = full_vector.flatten()
                        
                        vectors.append(full_vector)
                        
                        # 保存ID映射（使用当前样本号）
                        self.id_mapping[len(vectors) - 1] = local_idx
                    except Exception as e:
                        print(f"向量拼接失败，样本 {original_idx}: {e}")
                        print(f"向量形状: {[v.shape for v in vector_parts]}")
                
                pbar.set_postfix({"已处理": len(vectors), "当前样本": original_idx})
                pbar.update(1)
        
        return vectors
    
    def _get_category_weight(self, category, encoded_vector_length):
        """根据实际编码长度动态计算类别权重"""
        from .rag_weights import calculate_category_weight
        return calculate_category_weight(category, encoded_vector_length)
    
    def _create_faiss_index(self, vectors, index_type):
        """创建FAISS索引"""
        # 确保vectors是numpy数组
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        
        dimension = vectors.shape[1]
        vectors = vectors.astype('float32')
        
        if index_type == 'IVFFlat':
            # 使用IVFFlat平衡精度和速度
            quantizer = faiss.IndexFlatIP(dimension)  # 距离量化器(Quantizer)选择 内积相似度（通常用于余弦相似度，需提前标准化向量）
            # self.index = faiss.IndexIVFFlat(quantizer, dimension, min(1000, len(vectors) // 10))
            self.index = faiss.IndexIVFFlat(quantizer, dimension, min(1000, len(vectors) // 40))
            self.index.train(vectors)
            self.index.add(vectors)
            # self.index.nprobe = min(50, len(vectors) // 20)
            self.index.nprobe = min(500, len(vectors) // 60)  # 搜索的聚类数量
        elif index_type == 'Flat':
            # 使用Flat索引保证最高精度
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(vectors)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # 验证索引构建
        print(f"FAISS索引信息:")
        print(f"  - 向量维度: {dimension}")
        print(f"  - 索引大小: {self.index.ntotal}")
        print(f"  - 索引类型: {type(self.index).__name__}")
        
        # 测试索引是否正常工作
        test_query = vectors[0:1]  # 取第一个向量作为测试查询
        test_similarities, test_indices = self.index.search(test_query, min(5, len(vectors)))
        print(f"  - 测试查询相似度范围: {test_similarities[0].min():.3f} - {test_similarities[0].max():.3f}")
        print(f"  - 测试查询返回数量: {len(test_indices[0])}")
    
    def _save_mappings(self, df):
        """保存映射关系到文件"""
        mapping_data = {
            'id_mapping': self.id_mapping,
            'feature_dimensions': self.feature_dimensions,
            'feature_weights': self._get_features_default_weights(),
            'total_records': len(df),
            'dataset_name': self.dataset_name
        }
        
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(mapping_data, f)
    
    def save_index(self, index_path=None):
        """保存FAISS索引"""
        if self.index is not None:
            save_path = index_path or self.database_path
            faiss.write_index(self.index, save_path)
            self.database_path = save_path
            print(f"FAISS索引已保存到: {save_path}")
        else:
            raise ValueError("索引尚未构建，请先调用build_database方法")
    
    def load_database(self, index_path=None, mapping_path=None):
        """加载FAISS数据库"""
        try:
            # 使用默认路径
            index_path = index_path or self.database_path
            mapping_path = mapping_path or self.mapping_path
            encoder_path = self.encoder_path
            
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 加载映射关系
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                self.id_mapping = mapping_data['id_mapping']
                self.feature_weights = mapping_data['feature_weights']
            
            # 加载编码器
            self.encoder.load_encoder(encoder_path)
            
            self.database_path = index_path
            print(f"FAISS数据库加载成功，包含 {len(self.id_mapping)} 条记录")
            return True
            
        except Exception as e:
            print(f"FAISS数据库加载失败: {e}")
            return False
    
    def get_database_info(self):
        """获取数据库信息"""
        if self.index is None:
            return "数据库未加载"
        
        info = {
            'total_records': len(self.id_mapping),
            'vector_dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'database_path': self.database_path
        }
        return info
