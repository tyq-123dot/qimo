import numpy as np
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.config import config

class Vectorizer:
    """向量化器，负责文本到向量的转换"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.get('embedding.model_name', 'moka-ai/m3e-large')
        self.batch_size = config.get('embedding.batch_size', 32)
        self.normalize = config.get('embedding.normalize_embeddings', True)
        self.model = None
        self.dimension = None
    
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cache_folder = config.get('system.model_cache_dir', './models')
            
            print(f"加载嵌入模型: {self.model_name}")
            print(f"设备: {device}")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=cache_folder
            )
            
            # 获取模型维度
            test_embedding = self.model.encode(["测试文本"], normalize_embeddings=False)
            self.dimension = test_embedding.shape[1]
            
            print(f"模型加载成功，维度: {self.dimension}")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise VectorizationError(f"初始化模型失败: {str(e)}")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            np.ndarray: 嵌入向量矩阵
        """
        if not texts:
            return np.array([])
        
        if self.model is None:
            self.initialize()
        
        embeddings = []
        
        # 分批处理
        if show_progress:
            iterator = tqdm(range(0, len(texts), self.batch_size), desc="向量化")
        else:
            iterator = range(0, len(texts), self.batch_size)
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            if not batch:
                continue
            
            try:
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
            except Exception as e:
                raise VectorizationError(f"向量化批次 {i//self.batch_size} 失败: {str(e)}")
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def embed_query(self, query: str) -> np.ndarray:
        """嵌入单个查询"""
        if self.model is None:
            self.initialize()
        
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embedding[0]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.dimension is None:
            self.initialize()
        return self.dimension


class VectorizationError(Exception):
    """向量化异常"""
    pass