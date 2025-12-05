import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.config import config

class FAISSVectorStore:
    """FAISS向量数据库管理器"""
    
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path or config.get('system.vector_store_dir', './faiss_db')
        self.index_name = config.get('system.index_name', 'default')
        self.use_gpu = config.get('faiss.use_gpu', True)
        self.search_k = config.get('faiss.search_k', 5)
        
        self.index = None
        self.metadata = []
        
        # 创建目录
        os.makedirs(self.index_path, exist_ok=True)
    
    def build_index(self, embeddings: np.ndarray, metadatas: List[Dict]) -> bool:
        """
        构建向量索引
        
        Args:
            embeddings: 向量矩阵
            metadatas: 元数据列表
            
        Returns:
            bool: 是否成功
        """
        if embeddings.size == 0:
            print("警告: 没有向量可以构建索引")
            return False
        
        dimension = embeddings.shape[1]
        
        try:
            # 创建索引
            self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
            
            # 转移到GPU（如果可用）
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    print("使用GPU加速FAISS索引")
                except Exception as e:
                    print(f"GPU加速失败，使用CPU: {e}")
            
            # 添加向量
            self.index.add(embeddings.astype(np.float32))
            self.metadata = metadatas
            
            # 添加时间戳
            for meta in self.metadata:
                meta['indexed_at'] = datetime.now().isoformat()
            
            print(f"索引构建完成，包含 {len(metadatas)} 个向量")
            return True
            
        except Exception as e:
            raise IndexBuildError(f"构建索引失败: {str(e)}")
    
    def save_index(self, name: Optional[str] = None) -> bool:
        """保存索引到磁盘"""
        if self.index is None:
            print("警告: 没有索引可以保存")
            return False
        
        name = name or self.index_name
        index_file = os.path.join(self.index_path, f"{name}.index")
        meta_file = os.path.join(self.index_path, f"{name}_meta.pkl")
        
        try:
            # 如果是GPU索引，先转换到CPU
            if faiss.get_num_gpus() > 0 and hasattr(self.index, 'device'):  # GPU索引
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_file)
            else:  # CPU索引
                faiss.write_index(self.index, index_file)
            
            # 保存元数据
            with open(meta_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            print(f"索引已保存: {index_file}")
            return True
            
        except Exception as e:
            raise IndexSaveError(f"保存索引失败: {str(e)}")
    
    def load_index(self, name: Optional[str] = None) -> bool:
        """从磁盘加载索引"""
        name = name or self.index_name
        index_file = os.path.join(self.index_path, f"{name}.index")
        meta_file = os.path.join(self.index_path, f"{name}_meta.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            print(f"索引文件不存在: {index_file}")
            return False
        
        try:
            # 加载索引
            self.index = faiss.read_index(index_file)
            
            # 转移到GPU（如果可用）
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    print(f"GPU加速失败: {e}")
            
            # 加载元数据
            with open(meta_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"索引加载成功，包含 {len(self.metadata)} 个向量")
            return True
            
        except Exception as e:
            raise IndexLoadError(f"加载索引失败: {str(e)}")
    
    def similarity_search(self, query_vector: np.ndarray, k: Optional[int] = None) -> List[Dict]:
        """
        相似度搜索
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if self.index is None:
            raise ValueError("索引未初始化")
        
        k = k or self.search_k
        
        # 确保查询向量是2D数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 搜索
        distances, indices = self.index.search(
            query_vector.astype(np.float32), k
        )
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    'metadata': self.metadata[idx],
                    'score': float(distances[0][i]),
                    'rank': i + 1
                })
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if self.index is None:
            return {'status': '未初始化'}
        
        return {
            'status': '已加载',
            'total_vectors': self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.metadata),
            'dimension': self.index.d if hasattr(self.index, 'd') else '未知',
            'metadata_count': len(self.metadata)
        }


class IndexBuildError(Exception):
    """索引构建异常"""
    pass

class IndexSaveError(Exception):
    """索引保存异常"""
    pass

class IndexLoadError(Exception):
    """索引加载异常"""
    pass