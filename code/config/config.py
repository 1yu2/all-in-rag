"""
已授权模型：
1、deepseek-V3接口
4、Qwen3-235B-A22B接口
7、Qwen3-Embedding-8B模型接口
10、Qwen2.5-VL-72B模型接口
"""
import os

# 临时方案：直接读取 .env 文件
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

# 简化的配置类
class Settings:
    def __init__(self):
        # ============ Embedding 模型配置 (千问) ============
        self.EMBEDDING_API_KEY = os.getenv("API_KEY")
        self.EMBEDDING_MODEL_NAME = "Qwen3-Embedding-8B"
        self.EMBEDDING_MODEL_URL = "http://202.104.121.181:30012/gateway/ti/qwen3-embedding/v1/embeddings"
        self.EMBEDDING_SZC_API_KEY = "53de6400f8bf4c1ca2"

        # ============ LLM 模型配置 (Qwen) ============
        self.LLM_API_KEY = os.getenv("API_KEY")
        self.LLM_API_URL = "http://202.104.121.181:30012/gateway/ti/qwen3-235b/v1/chat/completions"
        self.LLM_MODEL_NAME = "Qwen3-235B"
        self.LLM_TEMPERATURE = 0.7
        self.LLM_TIMEOUT = 60

        # ============ VL 模型配置 (Qwen2.5-VL-72B) ============
        self.VL_API_KEY = os.getenv("API_KEY")
        self.VL_API_URL = "http://202.104.121.181:30012/gateway/ti/qwen2_5-vl-72b/v1/chat/completions"
        self.VL_MODEL_NAME = "Qwen2.5-VL-72B"
        self.VL_TEMPERATURE = 0.7
        self.VL_TIMEOUT = 60

settings = Settings()