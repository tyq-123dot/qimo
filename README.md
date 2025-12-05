# 中文RAG文档处理模块

基于RAG架构的中文文档处理与向量库构建模块，负责文档加载、文本分割、向量化和向量存储。

## 功能特性

- 📄 **多格式文档支持**: PDF、TXT、MD、DOCX
- 🇨🇳 **中文优化**: 针对中文文本的语义分割
- 🚀 **高性能向量化**: 基于m3e-large模型
- 💾 **本地向量存储**: 使用FAISS构建本地向量数据库
- 🔄 **增量更新**: 支持文档增量添加
- ⚙️ **配置驱动**: YAML配置文件管理

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/chinese-rag-document-processor.git
cd chinese-rag-document-processor

2. 安装依赖
```bash
pip install -r requirements.txt

3. 配置环境
```bash
# 编辑配置文件（可选）
cp config.example.yaml config.yaml
