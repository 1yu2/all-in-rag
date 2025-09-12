import requests
from typing import List
import logging
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接导入，避免 IDE 静态分析问题
try:
    from config.config import settings
except ImportError:
    # 如果导入失败，尝试其他路径
    import importlib.util
    config_path = os.path.join(project_root, 'config', 'config.py')
    if os.path.exists(config_path):
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        settings = config_module.settings
    else:
        raise ImportError(f"Cannot find config file at {config_path}")

logger = logging.getLogger(__name__)


class ThirdEmbedding:
    """
    统一的第三方 Embedding 服务客户端
    支持从统一配置自动获取设置，也支持手动传入参数
    """
    _instances = {}

    def __init__(self, api_key: str = None, model_name: str = None, base_url: str = None, **kwargs):
        """
        初始化 thirdEmbedding 客户端

        Args:
            api_key: API密钥，如果不提供则从配置读取
            model_name: 模型名称，如果不提供则从配置读取
            base_url: API地址，如果不提供则从配置读取
            **kwargs: 其他参数
        """
        # 获取默认配置
        self.api_key = settings.EMBEDDING_SZC_API_KEY
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.base_url = settings.EMBEDDING_MODEL_URL
        self.dimensions = 4096
        self.timeout = 30

    @classmethod
    def get_instance(cls, **kwargs):
        """
        获取单例实例
        """
        # 生成实例键
        instance_key = "default"
        if instance_key not in cls._instances:
            cls._instances[instance_key] = cls(**kwargs)
        return cls._instances[instance_key]

    def send_request(self, data):
        """发送请求到 Embedding API，并返回嵌入向量"""
        headers = {
            'szc-api-key': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            # 发送 POST 请求到 Embedding API
            response = requests.post(self.base_url, headers=headers, json=data, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()  # 返回成功响应的 JSON 数据
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
        except requests.exceptions.Timeout:
            error_msg = f"API请求超时 (超过 {self.timeout} 秒)"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入多个文档，并返回向量

        Args:
            texts: 要嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        data = {
            "model": self.model_name,
            "input": texts
        }

        try:
            result = self.send_request(data)
            # 提取嵌入向量数据
            embeddings = [item['embedding'] for item in result.get("data", [])]
            logger.info(f"成功嵌入 {len(embeddings)} 个文档")
            return embeddings
        except Exception as e:
            logger.error(f"文档嵌入失败: {str(e)}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        嵌入单个查询，并返回向量

        Args:
            query: 要嵌入的查询文本

        Returns:
            List[float]: 嵌入向量
        """
        if not query:
            return []

        data = {
            "model": self.model_name,
            "input": [query]
        }

        try:
            result = self.send_request(data)
            embedding = result.get("data", [])[0].get("embedding", [])
            logger.info(f"成功嵌入查询: {query[:50]}{'...' if len(query) > 50 else ''}")
            return embedding
        except Exception as e:
            logger.error(f"查询嵌入失败: {str(e)}")
            raise


embedding_client = ThirdEmbedding()

# 示例用法
def main():
    """示例：如何使用 thirdEmbedding 类"""
    print("=" * 50)

    try:
        embedding_client = ThirdEmbedding()

        # 嵌入多个文档
        texts = ["这是第一段测试文本。", "这是第二段测试文本。", "这是第三段测试文本。"]
        print(f"   - 输入文档数量: {len(texts)}")

        embeddings = embedding_client.embed_documents(texts)
        print(f"   - 输出向量数量: {len(embeddings)}")
        for emb in embeddings:
            print(f"   - <UNK>: {emb}")

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
