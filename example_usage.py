#!/usr/bin/env python3
"""
知识库构建模块使用示例
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_base import KnowledgeBaseBuilder, DocumentUploadManager

def main():
    """主函数"""
    print("=== 中文RAG文档处理模块示例 ===\n")
    
    # 1. 创建知识库构建器
    print("1. 初始化知识库构建器...")
    builder = KnowledgeBaseBuilder()
    
    # 2. 构建知识库
    print("\n2. 构建知识库...")
    result = builder.build_knowledge_base(rebuild=False)
    
    if result['success']:
        print(f"✓ {result['message']}")
        print(f"   统计信息: {result['stats']}")
        print(f"   耗时: {result['time_elapsed']:.2f}秒")
    else:
        print(f"✗ {result['message']}")
        # 可能需要先上传文档
        print("提示: 请先上传文档到 uploads/ 目录")
    
    # 3. 搜索示例
    print("\n3. 搜索测试...")
    queries = [
        "什么是机器学习？",
        "深度学习有哪些应用？",
        "神经网络的基本原理是什么？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = builder.search(query, k=3)
        
        if results:
            print(f"  找到 {len(results)} 个结果:")
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', '未知')
                score = result['score']
                print(f"  {i}. [{source}] 相似度: {score:.4f}")
        else:
            print("  未找到相关结果")
    
    # 4. 文档上传示例
    print("\n4. 文档上传管理器示例...")
    upload_manager = DocumentUploadManager()
    
    # 模拟上传文件
    print("创建测试文档...")
    test_content = "这是一个测试文档，用于演示文档上传功能。\n机器学习是人工智能的一个重要分支。"
    test_filename = "test_document.txt"
    
    upload_result = upload_manager.upload_document(
        test_content.encode('utf-8'),
        test_filename
    )
    
    if upload_result['success']:
        print(f"✓ {upload_result['message']}")
        
        # 重新构建知识库
        print("\n重新构建知识库（包含新文档）...")
        result = builder.build_knowledge_base(rebuild=True)
        
        if result['success']:
            print(f"✓ {result['message']}")
            print(f"   新统计信息: {result['stats']}")
        else:
            print(f"✗ {result['message']}")
    else:
        print(f"✗ {upload_result['message']}")
    
    print("\n=== 示例完成 ===")

if __name__ == "__main__":
    main()