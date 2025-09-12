import os
import sys
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from typing import List

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模块 - 使用绝对路径
llm_embedding_vl_path = os.path.join(project_root, 'llm_embedding_vl')
sys.path.insert(0, llm_embedding_vl_path)

try:
    import embedding
    # 创建实例
    embedding_client = embedding.embedding_client
    print(f"成功从 {llm_embedding_vl_path} 导入 embedding 模块")
except ImportError as e:
    print(f"导入失败: {e}")
    # 备用方案：使用相对路径
    import importlib.util

    embedding_spec = importlib.util.spec_from_file_location("embedding", os.path.join(llm_embedding_vl_path, "embedding.py"))
    embedding_module = importlib.util.module_from_spec(embedding_spec)
    embedding_spec.loader.exec_module(embedding_module)

    embedding_client = embedding_module.embedding_client
    print(f"使用备用方案成功导入模块")

class CustomLangChainEmbeddings:
    """自定义 LangChain 兼容的嵌入类，包装 llm_embedding_vl 中的 embedding_client"""

    def __init__(self, embedding_client):
        self.embedding_client = embedding_client
        self.model_name = "custom_embedding_model"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        return self.embedding_client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embedding_client.embed_query(text)

    def __call__(self, text: str) -> List[float]:
        """使对象可调用"""
        return self.embed_query(text)

# 创建自定义嵌入实例
embeddings = CustomLangChainEmbeddings(embedding_client)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)

loader = TextLoader("/Users/fishyuu/all-in-rag/data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
