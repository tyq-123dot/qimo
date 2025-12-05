import yaml
import os
from typing import Dict, Any

class Config:
    """配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            # 使用默认配置
            self._config = self._get_default_config()
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        
        # 创建必要的目录
        self._create_directories()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'system': {
                'upload_dir': './uploads',
                'model_cache_dir': './models',
                'vector_store_dir': './faiss_db',
                'index_name': 'default'
            },
            'text_splitter': {
                'chunk_size': 800,
                'chunk_overlap': 150,
                'separators': ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            },
            'embedding': {
                'model_name': 'moka-ai/m3e-large',
                'batch_size': 32,
                'normalize_embeddings': True
            },
            'faiss': {
                'use_gpu': True,
                'search_k': 5
            },
            'document': {
                'supported_extensions': ['.pdf', '.txt', '.md', '.docx'],
                'max_file_size_mb': 50
            }
        }
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.get('system.upload_dir'),
            self.get('system.model_cache_dir'),
            self.get('system.vector_store_dir')
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值，支持点分隔符"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """更新配置值"""
        keys = key.split('.')
        config = self._config
        
        # 遍历到倒数第二个key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置最后一个key的值
        config[keys[-1]] = value
    
    def save(self):
        """保存配置到文件"""
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

# 全局配置实例
config = Config()