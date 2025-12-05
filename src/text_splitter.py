import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.config import config

class ChineseSemanticSplitter:
    """中文语义文本分割器"""
    
    def __init__(self):
        self.chunk_size = config.get('text_splitter.chunk_size', 800)
        self.chunk_overlap = config.get('text_splitter.chunk_overlap', 150)
        self.separators = config.get('text_splitter.separators', [])
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._chinese_aware_length,
            is_separator_regex=False
        )
    
    def _chinese_aware_length(self, text: str) -> int:
        """
        中文字符长度计算，考虑中英文差异
        
        Args:
            text: 输入文本
            
        Returns:
            int: 调整后的长度
        """
        if not text:
            return 0
        
        # 中文字符每个算1，英文字符每2个算1
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        other_chars = re.findall(r'[^\u4e00-\u9fffa-zA-Z\s]', text)
        
        # 计算加权长度
        length = len(chinese_chars) + len(other_chars) + (len(english_chars) // 2 + len(english_chars) % 2)
        
        return length
    
    def split_documents(self, documents: List) -> List:
        """
        分割文档并添加元数据
        
        Args:
            documents: 文档对象列表
            
        Returns:
            List: 分割后的文本块列表
        """
        if not documents:
            return []
        
        # 分割文档
        chunks = self.splitter.split_documents(documents)
        
        # 为每个块添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            
            # 添加文本摘要（前50个字符）
            if len(chunk.page_content) > 50:
                chunk.metadata['summary'] = chunk.page_content[:50] + "..."
            else:
                chunk.metadata['summary'] = chunk.page_content
        
        print(f"文档分割完成，得到 {len(chunks)} 个文本块")
        return chunks
    
    def split_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        分割单个文本
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            List[dict]: 分割后的文本块列表
        """
        if not text:
            return []
        
        # 使用splitter分割文本
        chunks = self.splitter.split_text(text)
        
        # 构建结果
        results = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            results.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return results