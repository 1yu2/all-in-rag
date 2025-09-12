import os
import asyncio
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

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

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 构建向量存储 - 使用llm_embedding_vl中的embedding客户端
vectorstore = InMemoryVectorStore(embedding_client)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 使用llm_embedding_vl中的LLM客户端
async def get_answer():
    """获取LLM回答"""
    formatted_prompt = prompt.format(question=question, context=docs_content)
    response = await call_llm_async(formatted_prompt)
    return response

# 运行异步函数
if __name__ == "__main__":
    answer = asyncio.run(get_answer())
    print(answer)