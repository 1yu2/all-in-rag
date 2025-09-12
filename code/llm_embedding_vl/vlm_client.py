"""
异步 VLM 视觉语言模型客户端
========================

提供异步调用视觉语言模型的简化接口，支持流式和非流式调用。

基本使用方法：
============

1. 非流式调用：
   from app.llm.vlm_client import call_vlm_async
   response = await call_vlm_async("描述这张图片", images=["path/to/image.jpg"])

2. 流式调用：
   from app.llm.vlm_client import call_vlm_stream_async
   async for chunk in call_vlm_stream_async("分析图片内容", images=["image.jpg"]):
       print(chunk, end='')

3. 自定义参数：
   response = await call_vlm_async(
       "描述图片",
       images=["image.jpg"],
       temperature=0.5,     # 控制随机性
       max_tokens=1024      # 最大输出长度
   )

图片支持：
=========
支持多种图片输入格式：
- 文件路径字符串: "path/to/image.jpg"
- 字节数据: bytes
"""
import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Dict, List, Union
import requests
import sys
import os

logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置
try:
    from code.config import settings
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


def _get_vlm_config() -> dict:
    """获取 VLM 模型配置"""
    return {
        "api_url": settings.VL_API_URL,
        "api_key": settings.VL_API_KEY,
        "model_name": settings.VL_MODEL_NAME,
        "temperature": settings.VL_TEMPERATURE,
        "timeout": settings.VL_TIMEOUT
    }


async def _encode_image_to_base64(image_input: Union[str, bytes]) -> str:
    """异步将图片编码为 base64 字符串"""
    try:
        if isinstance(image_input, str):
            # 如果是文件路径，异步读取文件
            loop = asyncio.get_event_loop()
            with open(image_input, "rb") as f:
                image_data = await loop.run_in_executor(None, f.read)
        elif isinstance(image_input, bytes):
            image_data = image_input
        else:
            raise ValueError(f"不支持的图片类型: {type(image_input)}，仅支持文件路径或字节数据")

        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"图片编码失败: {e}")
        raise


async def _make_request_payload(prompt: str, images: List[Union[str, bytes]],
                                stream: bool = False, **kwargs) -> Dict:
    """构建请求负载"""
    config = _get_vlm_config()

    # 处理图片
    content = [{"type": "text", "text": prompt}]

    for img in images:
        img_b64 = await _encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    payload = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        "temperature": kwargs.get("temperature", config["temperature"]),
    }

    # 只有在提供了max_tokens参数时才添加
    if "max_tokens" in kwargs:
        payload["max_tokens"] = kwargs["max_tokens"]

    return payload


def _extract_content(data: Dict) -> str:
    """提取响应内容"""
    if 'choices' in data and data['choices']:
        choice = data['choices'][0]
        if 'message' in choice:
            return choice['message'].get('content', '')
        elif 'delta' in choice:
            return choice['delta'].get('content', '')
    return ''


async def call_vlm_async(prompt: str, images: List[Union[str, bytes]], **kwargs) -> str:
    """异步调用 VLM 视觉语言模型（非流式）

    Args:
        prompt: 输入提示词
        images: 图片列表（文件路径或字节数据）
        **kwargs: 可选参数（temperature, max_tokens 等）

    Returns:
        str: 模型回复内容

    Raises:
        ValueError: API调用失败时抛出
    """
    if not images:
        raise ValueError("VLM调用必须提供至少一张图片")

    config = _get_vlm_config()
    payload = await _make_request_payload(prompt, images, stream=False, **kwargs)

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
        logger.error(f"VLM API调用失败: {e}")
        raise ValueError(f"网络请求失败: {e}")
    except Exception as e:
        logger.error(f"VLM调用异常: {e}")
        raise ValueError(f"API调用失败: {e}")


async def call_vlm_stream_async(prompt: str, images: List[Union[str, bytes]], **kwargs) -> AsyncGenerator[str, None]:
    """异步调用 VLM 视觉语言模型（流式）

    Args:
        prompt: 输入提示词
        images: 图片列表（文件路径或字节数据）
        **kwargs: 可选参数（temperature, max_tokens 等）

    Yields:
        str: 模型回复的文本块

    Raises:
        ValueError: API调用失败时抛出
    """
    if not images:
        raise ValueError("VLM调用必须提供至少一张图片")

    config = _get_vlm_config()
    payload = await _make_request_payload(prompt, images, stream=True, **kwargs)

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
        logger.error(f"VLM流式API调用失败: {e}")
        raise ValueError(f"网络请求失败: {e}")
    except Exception as e:
        logger.error(f"VLM流式调用异常: {e}")
        raise ValueError(f"流式API调用失败: {e}")


async def call_vlm_batch_async(prompt_image_pairs: List[tuple], **kwargs) -> List[str]:
    """批量异步调用 VLM 视觉语言模型

    Args:
        prompt_image_pairs: [(prompt1, [images1]), (prompt2, [images2]), ...]
        **kwargs: 可选参数（temperature, max_tokens 等）

    Returns:
        List[str]: 模型回复列表
    """
    tasks = [call_vlm_async(prompt, images, **kwargs)
             for prompt, images in prompt_image_pairs]
    return await asyncio.gather(*tasks)


def get_vlm_config() -> dict:
    """获取 VLM 配置信息

    Returns:
        dict: 配置信息
    """
    return _get_vlm_config()


# 示例用法
async def main():
    """测试VLM模型是否可用"""
    print("🔍 测试VLM模型连接状态...")

    try:
        config = _get_vlm_config()

        # 直接构建简单请求
        payload = {
            "model": config["model_name"],
            "messages": [{"role": "user", "content": "测试连接"}],
            "stream": False,
            "temperature": 0.7
        }

        headers = {
            "Content-Type": "application/json",
            "szc-api-key": config["api_key"]
        }

        # 直接发送POST请求
        loop = asyncio.get_event_loop()
        def _sync_request():
            response = requests.post(
                config["api_url"],
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        result = await loop.run_in_executor(None, _sync_request)

        # 检查响应
        if result and 'choices' in result and result['choices']:
            print("✅ VLM模型可用")
            return True
        else:
            print("❌ VLM模型响应异常")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def run_tests():
    """运行VLM模型可用性测试"""
    try:
        result = asyncio.run(main())
        return 0 if result else 1
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return 1


if __name__ == "__main__":
    exit(run_tests())
