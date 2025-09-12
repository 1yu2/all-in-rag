import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import List, Optional
import sys
# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模块 - 使用绝对路径
llm_embedding_vl_path = os.path.join(project_root, 'llm_embedding_vl')
sys.path.insert(0, llm_embedding_vl_path)

try:
    import embedding
    import llm_client
    # 创建实例
    embedding_client = embedding.embedding_client
    call_llm_async = llm_client.call_llm_async
    print(f"成功从 {llm_embedding_vl_path} 导入模块")
except ImportError as e:
    print(f"导入失败: {e}")
    # 备用方案：使用相对路径
    import importlib.util

    embedding_spec = importlib.util.spec_from_file_location("embedding", os.path.join(llm_embedding_vl_path, "embedding.py"))
    embedding_module = importlib.util.module_from_spec(embedding_spec)
    embedding_spec.loader.exec_module(embedding_module)

    llm_spec = importlib.util.spec_from_file_location("llm_client", os.path.join(llm_embedding_vl_path, "llm_client.py"))
    llm_module = importlib.util.module_from_spec(llm_spec)
    llm_spec.loader.exec_module(llm_module)

    embedding_client = embedding_module.embedding_client
    call_llm_async = llm_module.call_llm_async

    print(f"使用备用方案成功导入模块")

class CustomEmbedding(BaseEmbedding):
    """自定义嵌入类，包装llm_embedding_vl中的embedding_client"""

    def __init__(self, embedding_client, **kwargs):
        super().__init__(**kwargs)
        self._embedding_client = embedding_client

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self._embedding_client.embed_query(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的嵌入向量"""
        return self._embedding_client.embed_documents(texts)

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询的嵌入向量"""
        return self._embedding_client.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询的嵌入向量"""
        return self._embedding_client.embed_query(query)

class CustomLLMWrapper(CustomLLM):
    """自定义LLM类，包装llm_embedding_vl中的call_llm_async"""

    def __init__(self, call_llm_func, **kwargs):
        super().__init__(**kwargs)
        self._call_llm_func = call_llm_func

    @property
    def metadata(self):
        from llama_index.core.llms import LLMMetadata
        return LLMMetadata(context_window=4096, num_output=1024)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> str:
        """同步完成方法"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._call_llm_func(prompt, **kwargs))
        finally:
            loop.close()

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """异步完成方法"""
        return await self._call_llm_func(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        """流式完成方法 - 不支持流式，返回完整结果"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._call_llm_func(prompt, **kwargs))
            from llama_index.core.llms import CompletionResponse
            yield CompletionResponse(text=result)
        finally:
            loop.close()

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs):
        """异步流式完成方法 - 不支持流式，返回完整结果"""
        result = await self._call_llm_func(prompt, **kwargs)
        from llama_index.core.llms import CompletionResponse
        yield CompletionResponse(text=result)

load_dotenv()

# 创建自定义嵌入和LLM实例
custom_embedding = CustomEmbedding(embedding_client)
custom_llm = CustomLLMWrapper(call_llm_async)

docs = SimpleDirectoryReader(input_files=["/Users/fishyuu/all-in-rag/data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 使用自定义嵌入创建索引
index = VectorStoreIndex.from_documents(docs, embed_model=custom_embedding)

# 使用自定义LLM创建查询引擎
query_engine = index.as_query_engine(llm=custom_llm)

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))