import os
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from .document_processor import DocumentProcessor
from .text_splitter import ChineseSemanticSplitter
from .vectorizer import Vectorizer
from .vector_store import FAISSVectorStore
from utils.config import config

class KnowledgeBaseBuilder:
    """知识库构建器，协调整个处理流程"""
    
    def __init__(self):
        self.upload_dir = config.get('system.upload_dir', './uploads')
        self.document_processor = DocumentProcessor()
        self.text_splitter = ChineseSemanticSplitter()
        self.vectorizer = Vectorizer()
        self.vector_store = FAISSVectorStore()
        
        # 创建目录
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def build_knowledge_base(self, rebuild: bool = False) -> Dict[str, Any]:
        """
        构建或加载知识库
        
        Args:
            rebuild: 是否强制重建
            
        Returns:
            Dict: 构建结果
        """
        start_time = time.time()
        
        try:
            # 检查是否已存在索引且不需要重建
            if not rebuild and self.vector_store.load_index():
                print("✓ 加载现有知识库")
                stats = self.vector_store.get_index_stats()
                return {
                    'success': True,
                    'message': '知识库加载成功',
                    'stats': stats,
                    'time_elapsed': time.time() - start_time
                }
            
            print("开始构建知识库...")
            
            # 1. 加载文档
            documents = self._load_all_documents()
            if not documents:
                return {
                    'success': False,
                    'message': '没有找到可处理的文档',
                    'time_elapsed': time.time() - start_time
                }
            
            print(f"✓ 加载 {len(documents)} 个文档片段")
            
            # 2. 分割文档
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                return {
                    'success': False,
                    'message': '文档分割失败',
                    'time_elapsed': time.time() - start_time
                }
            
            print(f"✓ 分割为 {len(chunks)} 个文本块")
            
            # 3. 提取文本和元数据
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # 4. 向量化
            print("开始向量化...")
            embeddings = self.vectorizer.embed_texts(texts)
            if embeddings.size == 0:
                return {
                    'success': False,
                    'message': '向量化失败',
                    'time_elapsed': time.time() - start_time
                }
            
            print(f"✓ 向量化完成，维度: {embeddings.shape}")
            
            # 5. 构建索引
            self.vector_store.build_index(embeddings, metadatas)
            
            # 6. 保存索引
            self.vector_store.save_index()
            
            time_elapsed = time.time() - start_time
            stats = self.vector_store.get_index_stats()
            
            print(f"✓ 知识库构建完成，耗时: {time_elapsed:.2f}秒")
            
            return {
                'success': True,
                'message': '知识库构建成功',
                'stats': stats,
                'time_elapsed': time_elapsed
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'知识库构建失败: {str(e)}',
                'time_elapsed': time.time() - start_time
            }
    
    def _load_all_documents(self) -> List:
        """加载上传目录中的所有文档"""
        all_documents = []
        
        # 检查上传目录是否存在
        if not os.path.exists(self.upload_dir):
            print(f"上传目录不存在: {self.upload_dir}")
            return all_documents
        
        # 遍历上传目录
        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            
            # 只处理支持的文件
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.document_processor.supported_extensions:
                continue
            
            # 检查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            max_size = config.get('document.max_file_size_mb', 50)
            if file_size_mb > max_size:
                print(f"跳过文件（超过大小限制）: {filename} ({file_size_mb:.1f}MB)")
                continue
            
            try:
                print(f"加载文档: {filename}")
                documents = self.document_processor.load_document(file_path)
                all_documents.extend(documents)
                print(f"  ✓ 成功加载 {len(documents)} 个文档片段")
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
                continue
        
        return all_documents
    
    def search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        在知识库中搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        if self.vector_store.index is None:
            # 尝试加载索引
            if not self.vector_store.load_index():
                return []
        
        try:
            # 向量化查询
            query_vector = self.vectorizer.embed_query(query)
            
            # 搜索
            results = self.vector_store.similarity_search(query_vector, k)
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if self.vector_store.index is None:
            if not self.vector_store.load_index():
                return {'status': '未初始化'}
        
        return self.vector_store.get_index_stats()


class DocumentUploadManager:
    """文档上传管理器"""
    
    def __init__(self, upload_dir: Optional[str] = None):
        self.upload_dir = upload_dir or config.get('system.upload_dir', './uploads')
        self.processed_files = set()
        
        # 创建上传目录
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def upload_document(self, file_obj, filename: str) -> Dict[str, Any]:
        """
        上传文档
        
        Args:
            file_obj: 文件对象
            filename: 原始文件名
            
        Returns:
            Dict: 上传结果
        """
        try:
            # 生成唯一文件名
            unique_filename = self._generate_unique_filename(filename)
            file_path = os.path.join(self.upload_dir, unique_filename)
            
            # 保存文件
            if hasattr(file_obj, 'read'):
                content = file_obj.read()
            else:
                content = file_obj
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # 记录处理状态
            self.processed_files.add(unique_filename)
            
            return {
                'success': True,
                'message': f"文档上传成功: {unique_filename}",
                'filename': unique_filename,
                'file_path': file_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"文档上传失败: {str(e)}"
            }
    
    def _generate_unique_filename(self, filename: str) -> str:
        """生成唯一文件名"""
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        
        # 清理文件名中的特殊字符
        clean_name = re.sub(r'[^\w\-\.]', '_', name)
        
        return f"{clean_name}_{timestamp}_{unique_id}{ext}"
    
    def clear_upload_dir(self) -> Dict[str, Any]:
        """清空上传目录"""
        try:
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            self.processed_files.clear()
            
            return {
                'success': True,
                'message': '上传目录已清空'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'清空上传目录失败: {str(e)}'
            }