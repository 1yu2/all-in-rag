"""
异步 LLM 文本模型客户端
===================

提供异步调用大语言模型的简化接口，支持流式和非流式调用。

基本使用方法：
============

1. 非流式调用：
   from app.llm.llm_client import call_llm_async
   response = await call_llm_async("你好，请介绍一下深圳")

2. 流式调用：
   from app.llm.llm_client import call_llm_stream_async
   async for chunk in call_llm_stream_async("写一篇关于深圳的文章"):
       print(chunk, end='')

3. 自定义参数：
   response = await call_llm_async(
       "你好",
       temperature=0.5,     # 控制随机性
       max_tokens=1024      # 最大输出长度
   )
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, List
import requests
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# 导入配置
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


def _get_llm_config() -> dict:
    """获取 LLM 模型配置"""
    return {
        "api_url": settings.LLM_API_URL,
        "api_key": settings.LLM_API_KEY,
        "model_name": settings.LLM_MODEL_NAME,
        "temperature": settings.LLM_TEMPERATURE,
        "timeout": settings.LLM_TIMEOUT
    }


def _make_request_payload(prompt: str, stream: bool = False, **kwargs) -> Dict:
    """构建请求负载"""
    config = _get_llm_config()

    return {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": kwargs.get("temperature", config["temperature"]),
    }


def _extract_content(data: Dict) -> str:
    """提取响应内容"""
    if 'choices' in data and data['choices']:
        choice = data['choices'][0]
        if 'message' in choice:
            return choice['message'].get('content', '')
        elif 'delta' in choice:
            return choice['delta'].get('content', '')
    return ''


async def call_llm_async(prompt: str, **kwargs) -> str:
    """异步调用 LLM 文本模型（非流式）

    Args:
        prompt: 输入提示词
        **kwargs: 可选参数（temperature, max_tokens 等）

    Returns:
        str: 模型回复内容

    Raises:
        ValueError: API调用失败时抛出
    """
    config = _get_llm_config()
    payload = _make_request_payload(prompt, stream=False, **kwargs)

    headers = {
        "Content-Type": "application/json",
        "szc-api-key": config["api_key"]
    }

    try:
        # 使用 asyncio 在线程池中运行同步请求
        loop = asyncio.get_event_loop()

        def _sync_request():
            response = requests.post(
                config["api_url"],
                json=payload,
                headers=headers,
                timeout=config["timeout"]
            )
            response.raise_for_status()
            return response.json()

        result = await loop.run_in_executor(None, _sync_request)
        content = _extract_content(result)
        if not content:
            raise ValueError("API返回空内容")

        return content

    except requests.RequestException as e:
        logger.error(f"LLM API调用失败: {e}")
        raise ValueError(f"网络请求失败: {e}")
    except Exception as e:
        logger.error(f"LLM调用异常: {e}")
        raise ValueError(f"API调用失败: {e}")


async def call_llm_stream_async(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
    """异步调用 LLM 文本模型（流式）

    Args:
        prompt: 输入提示词
        **kwargs: 可选参数（temperature, max_tokens 等）

    Yields:
        str: 模型回复的文本块

    Raises:
        ValueError: API调用失败时抛出
    """
    config = _get_llm_config()
    payload = _make_request_payload(prompt, stream=True, **kwargs)

    headers = {
        "Content-Type": "application/json",
        "szc-api-key": config["api_key"]
    }

    try:
        # 使用队列在线程和协程之间传递数据
        import queue
        import threading

        chunk_queue = queue.Queue()
        exception_holder = [None]

        def _sync_stream_request():
            try:
                response = requests.post(
                    config["api_url"],
                    json=payload,
                    headers=headers,
                    timeout=config["timeout"],
                    stream=True
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data:"):
                        continue

                    chunk_data = line_str[5:].strip()
                    if chunk_data == "[DONE]":
                        break

                    try:
                        data = json.loads(chunk_data)
                        content = _extract_content(data)
                        if content:
                            chunk_queue.put(content)
                    except json.JSONDecodeError:
                        continue

                chunk_queue.put(None)  # 结束标记
            except Exception as e:
                exception_holder[0] = e
                chunk_queue.put(None)

        # 在线程中运行同步流式请求
        thread = threading.Thread(target=_sync_stream_request)
        thread.start()

        # 异步获取数据
        while True:
            # 非阻塞检查队列
            try:
                chunk = chunk_queue.get_nowait()
                if chunk is None:  # 结束标记
                    break
                yield chunk
            except queue.Empty:
                # 队列为空，让出控制权
                await asyncio.sleep(0.01)
                continue

        # 等待线程结束
        thread.join()

        # 检查是否有异常
        if exception_holder[0]:
            raise exception_holder[0]

    except requests.RequestException as e:
        logger.error(f"LLM流式API调用失败: {e}")
        raise ValueError(f"网络请求失败: {e}")
    except Exception as e:
        logger.error(f"LLM流式调用异常: {e}")
        raise ValueError(f"流式API调用失败: {e}")


async def call_llm_batch_async(prompts: List[str], **kwargs) -> List[str]:
    """批量异步调用 LLM 文本模型

    Args:
        prompts: 提示词列表
        **kwargs: 可选参数（temperature, max_tokens 等）

    Returns:
        List[str]: 模型回复列表
    """
    tasks = [call_llm_async(prompt, **kwargs) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def test_llm_connection() -> bool:
    """测试 LLM 连接是否正常

    Returns:
        bool: 连接是否正常
    """
    try:
        response = await call_llm_async("测试连接")
        return len(response) > 0
    except Exception as e:
        logger.error(f"LLM连接测试失败: {e}")
        return False


def get_llm_config() -> dict:
    """获取 LLM 配置信息

    Returns:
        dict: 配置信息
    """
    return _get_llm_config()

# 示例用法
async def main():
    """测试LLM模型是否可用"""
    print("🔍 测试LLM模型连接状态...")

    try:
        is_available = await test_llm_connection()
        if is_available:
            print("✅ LLM模型可用")
        else:
            print("❌ LLM模型不可用")
        return is_available
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def run_tests():
    """运行LLM模型可用性测试"""
    try:
        result = asyncio.run(main())
        return 0 if result else 1
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return 1


if __name__ == "__main__":
    exit(run_tests())
