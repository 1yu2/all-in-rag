import os
import sys
import asyncio
import numpy as np
from typing import Union, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模块 - 使用绝对路径
llm_embedding_vl_path = os.path.join(project_root, 'llm_embedding_vl')
sys.path.insert(0, llm_embedding_vl_path)

try:
    import llm_client
    import vlm_client
    call_llm_async = llm_client.call_llm_async
    call_vlm_async = vlm_client.call_vlm_async
    print(f"成功从 {llm_embedding_vl_path} 导入 LLM 和 VLM 模块")
except ImportError as e:
    print(f"导入失败: {e}")
    import importlib.util

    llm_spec = importlib.util.spec_from_file_location("llm_client", os.path.join(llm_embedding_vl_path, "llm_client.py"))
    llm_module = importlib.util.module_from_spec(llm_spec)
    llm_spec.loader.exec_module(llm_module)

    vlm_spec = importlib.util.spec_from_file_location("vlm_client", os.path.join(llm_embedding_vl_path, "vlm_client.py"))
    vlm_module = importlib.util.module_from_spec(vlm_spec)
    vlm_spec.loader.exec_module(vlm_module)

    call_llm_async = llm_module.call_llm_async
    call_vlm_async = vlm_module.call_vlm_async
    print(f"使用备用方案成功导入模块")

class VLMEmbeddingAdapter:
    """VLM模型适配器，提供类似嵌入模型的功能"""

    def __init__(self, call_llm_func, call_vlm_func, embedding_dim=768):
        self.call_llm_func = call_llm_func
        self.call_vlm_func = call_vlm_func
        self.embedding_dim = embedding_dim

    async def encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        description = await self.call_llm_func(f"请生成这段文本的语义特征向量描述：{text}")
        return self._text_to_vector(description)

    async def encode_image(self, image_path: str) -> np.ndarray:
        """编码图像为向量"""
        try:
            description = await self.call_vlm_func(f"请分析这张图像并生成其特征向量描述", images=[image_path])
            return self._text_to_vector(description)
        except Exception as e:
            print(f"图像编码错误: {e}")
            # 返回零向量作为备选
            return np.zeros(self.embedding_dim)

    async def encode_multimodal(self, image_path: str, text: str) -> np.ndarray:
        """编码多模态输入为向量"""
        try:
            description = await self.call_vlm_func(
                f"请结合文本'{text}'分析这张图像，生成综合特征向量描述",
                images=[image_path]
            )
            return self._text_to_vector(description)
        except Exception as e:
            print(f"多模态编码错误: {e}")
            # 返回零向量作为备选
            return np.zeros(self.embedding_dim)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """将文本描述转换为向量（简化版本）"""
        words = text.lower().split()
        vector = np.zeros(self.embedding_dim)

        for i, word in enumerate(words):
            if i < self.embedding_dim:
                hash_val = hash(word) % self.embedding_dim
                vector[hash_val] += 1

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    async def compare_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

async def main():
    """VLM多模态处理示例"""
    print("=== VLM多模态处理示例 ===")

    # 创建VLM适配器
    vlm_adapter = VLMEmbeddingAdapter(call_llm_async, call_vlm_async)

    # 编码示例
    try:
        text_emb = await vlm_adapter.encode_text("datawhale开源组织的logo")
        print(f"文本向量维度: {text_emb.shape}")
        print(f"文本向量示例 (前10个元素): {text_emb[:10]}")

        # 检查图像文件是否存在
        img_path = "/Users/fishyuu/all-in-rag/data/C3/imgs/datawhale01.png"
        if os.path.exists(img_path):
            img_emb = await vlm_adapter.encode_image(img_path)
            print(f"图像向量维度: {img_emb.shape}")
            print(f"图像向量示例 (前10个元素): {img_emb[:10]}")

            multi_emb = await vlm_adapter.encode_multimodal(img_path, "datawhale开源组织的logo")
            print(f"多模态向量维度: {multi_emb.shape}")
            print(f"多模态向量示例 (前10个元素): {multi_emb[:10]}")

            # 计算相似度
            sim_1 = await vlm_adapter.compare_similarity(img_emb, text_emb)
            sim_2 = await vlm_adapter.compare_similarity(img_emb, multi_emb)
            sim_3 = await vlm_adapter.compare_similarity(text_emb, multi_emb)

            print("\n=== 相似度计算结果 ===")
            print(f"图像 vs 文本: {sim_1:.4f}")
            print(f"图像 vs 多模态: {sim_2:.4f}")
            print(f"文本 vs 多模态: {sim_3:.4f}")
        else:
            print(f"图像文件不存在: {img_path}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())