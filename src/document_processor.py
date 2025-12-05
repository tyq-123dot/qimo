import os
import re
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

from utils.config import config

class DocumentProcessor:
    """文档处理器，统一管理不同格式的文档加载"""
    
    def __init__(self):
        self.supported_extensions = set(config.get('document.supported_extensions', []))
    
    def load_document(self, file_path: str) -> List:
        """
        根据文件扩展名选择合适的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            List: 文档对象列表
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        try:
            if ext == '.pdf':
                documents = self._load_pdf(file_path)
            elif ext == '.txt' or ext == '.md':
                documents = self._load_text(file_path)
            elif ext == '.docx':
                documents = self._load_docx(file_path)
            else:
                raise ValueError(f"未实现 {ext} 文件的加载器")
            
            # 清理文本并添加元数据
            for doc in documents:
                doc.page_content = self._clean_chinese_text(doc.page_content)
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['file_path'] = file_path
            
            return documents
            
        except Exception as e:
            raise DocumentLoadingError(f"加载文档失败: {str(e)}")
    
    def _load_pdf(self, file_path: str) -> List:
        """加载PDF文档"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_text(self, file_path: str) -> List:
        """加载文本文件"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List:
        """加载Word文档"""
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    
    def _clean_chinese_text(self, text: str) -> str:
        """清理中文文本中的特殊字符和格式"""
        if not text:
            return ""
        
        # 移除不可见字符
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 合并多个空白字符
        text = re.sub(r'\s+', ' ', text)
        # 处理中文标点与空格的组合
        text = re.sub(r'\s*([，。；：！？])\s*', r'\1 ', text)
        # 移除多余的空格
        text = text.strip()
        
        return text
    
    def batch_load_documents(self, directory: str) -> List:
        """批量加载目录中的所有文档"""
        all_documents = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    try:
                        documents = self.load_document(file_path)
                        all_documents.extend(documents)
                        print(f"✓ 成功加载: {filename} ({len(documents)}个片段)")
                    except Exception as e:
                        print(f"✗ 加载失败: {filename} - {str(e)}")
        
        return all_documents


class DocumentLoadingError(Exception):
    """文档加载异常"""
    pass