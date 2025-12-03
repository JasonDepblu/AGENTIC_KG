#%%
import os
from dotenv import load_dotenv
load_dotenv()

# from huggingface_hub import snapshot_download
#
# # 将 BAAI/bge-m3 模型下载到本地 "E:/models/BAAI/bge-m3" 目录
# local_dir = "/Users/depblu/Documents/GitHub/genQA/models/BAAI/bge-m3"
# repo_id = "BAAI/bge-m3"
#
# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     # 可以通过 max_workers 参数并行下载多个文件，加速过程
#     max_workers=8
# )
# print(f"模型已下载到: {local_dir}")

"""
Embedding model
"""
# from langchain_mistralai.embeddings import MistralAIEmbeddings

# embedder = MistralAIEmbeddings(
#     model="mistral-embed",
#     # api_key="...",
#     # other params...
# )

# from langchain_openai import OpenAIEmbeddings
# embedder_qwen = OpenAIEmbeddings(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="text-embedding-v3",
# )

# Qwen embedding model
from langchain_community.embeddings import DashScopeEmbeddings
embedder_qwen = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# Multi-modal embedding support
import dashscope
import base64
import json
from http import HTTPStatus
import logging
from typing import List, Dict, Union, Optional
import numpy as np
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 初始化tiktoken编码器
enc = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)

class MultiModalEmbeddings:
    """
    多模态嵌入类，支持文本和图片的向量化
    使用DashScope的text-embedding-v4和multimodal-embedding-v1模型
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化多模态嵌入器
        
        Args:
            api_key: DashScope API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set in environment variables")
        
        # 设置API密钥
        dashscope.api_key = self.api_key
        
        # 文本嵌入器（复用现有的）
        self.text_embedder = embedder_qwen
        
    def embed_text(self, text: str) -> List[float]:
        """
        嵌入单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本向量
        """
        try:
            return self.text_embedder.embed_query(text)
        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量列表
        """
        try:
            return self.text_embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"批量文本嵌入失败: {e}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionResetError, ConnectionError, ConnectionAbortedError))
    )
    def embed_image(self, image_data: Union[str, bytes], image_format: str = "png") -> Optional[List[float]]:
        """
        嵌入单个图片

        Args:
            image_data: 图片数据，可以是base64字符串或bytes
            image_format: 图片格式 (png, jpg, jpeg, bmp等)

        Returns:
            图片向量
        """
        try:
            # 处理输入数据格式
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode('utf-8')
            elif isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # 已经是完整的data URI
                    image_input = image_data
                    # 提取base64部分来检查大小
                    base64_part = image_data.split(',')[1] if ',' in image_data else image_data
                    base64_image = base64_part
                else:
                    # 假设是base64编码字符串
                    base64_image = image_data
                    image_input = f"data:image/{image_format};base64,{base64_image}"
            else:
                raise ValueError("Unsupported image data format")
            
            if 'image_input' not in locals():
                image_input = f"data:image/{image_format};base64,{base64_image}"
            
            # 检查图片大小（3MB限制）
            image_size_bytes = len(base64.b64decode(base64_image))
            image_size_mb = image_size_bytes / (1024 * 1024)
            
            if image_size_mb > 3.0:
                logger.warning(f"图片大小 {image_size_mb:.2f}MB 超过3MB限制，尝试压缩...")
                
                # 尝试压缩图片
                compressed_data = self._compress_image(base64_image, image_format)
                if compressed_data:
                    compressed_size_mb = len(base64.b64decode(compressed_data)) / (1024 * 1024)
                    logger.info(f"图片压缩成功: {image_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB")
                    base64_image = compressed_data
                    image_input = f"data:image/{image_format};base64,{base64_image}"
                else:
                    logger.error(f"图片压缩失败，无法处理大小为 {image_size_mb:.2f}MB 的图片")
                    return None
            
            # 调用多模态嵌入API
            input_data = [{'image': image_input}]
            
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=input_data
            )
            
            if resp.status_code == HTTPStatus.OK:
                # 提取嵌入向量
                embeddings = resp.output.get('embeddings', [])
                if embeddings:
                    return embeddings[0].get('embedding', [])
                else:
                    logger.error("API返回中未找到嵌入向量")
                    return None
            else:
                logger.error(f"多模态嵌入API调用失败: {resp.code} - {resp.message}")
                return None
                
        except Exception as e:
            logger.error(f"图片嵌入失败: {e}")
            return None

    def embed_images(self, images_data: List[Dict]) -> List[List[float]]:
        """
        批量嵌入图片（循环调用 embed_image）

        Args:
            images_data: 图片数据列表，每个元素包含 'data' 和 'format' 字段
                        例如：[{'data': base64_str, 'format': 'png'}, ...]

        Returns:
            图片向量列表，失败的图片返回零向量
        """
        embeddings = []
        for img_data in images_data:
            emb = self.embed_image(img_data['data'], img_data['format'])
            embeddings.append(emb if emb else [0.0] * 1024)
        return embeddings

    def _compress_image(self, base64_image: str, image_format: str, max_size_mb: float = 2.8) -> Optional[str]:
        """
        压缩图片到指定大小以下
        
        Args:
            base64_image: base64编码的图片
            image_format: 图片格式
            max_size_mb: 最大大小（MB）
            
        Returns:
            压缩后的base64图片字符串
        """
        try:
            from PIL import Image
            import io
            
            # 解码base64图片
            image_bytes = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(image_bytes))
            
            # 如果是RGBA，转换为RGB（JPEG不支持透明通道）
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = rgb_img
            
            # 尝试不同的压缩策略
            quality = 85
            scale = 1.0
            
            while quality > 20:
                # 缩放图片
                if scale < 1.0:
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    temp_img = img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    temp_img = img
                
                # 压缩图片
                output = io.BytesIO()
                temp_img.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_bytes = output.getvalue()
                
                # 检查大小
                size_mb = len(compressed_bytes) / (1024 * 1024)
                if size_mb <= max_size_mb:
                    return base64.b64encode(compressed_bytes).decode('utf-8')
                
                # 调整压缩参数
                quality -= 10
                if quality <= 20 and scale > 0.5:
                    scale -= 0.1
                    quality = 85
            
            logger.error("无法将图片压缩到目标大小")
            return None
            
        except Exception as e:
            logger.error(f"图片压缩失败: {e}")
            return None
    
    def embed_image_from_file(self, image_path: str) -> Optional[List[float]]:
        """
        从文件路径嵌入图片
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            图片向量
        """
        try:
            import os
            from pathlib import Path
            
            image_file = Path(image_path)
            if not image_file.exists():
                logger.error(f"图片文件不存在: {image_path}")
                return None
            
            # 获取文件格式
            image_format = image_file.suffix.lower().lstrip('.')
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            # 读取文件并编码
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            return self.embed_image(image_data, image_format)
            
        except Exception as e:
            logger.error(f"从文件嵌入图片失败: {e}")
            return None
    
    def embed_multimodal_content(self, content: Dict) -> Optional[List[float]]:
        """
        根据内容类型选择合适的嵌入方法
        
        Args:
            content: 内容字典，包含'type'和'data'字段
                    type可以是'text'或'image'
                    
        Returns:
            内容向量
        """
        content_type = content.get('type', '').lower()
        data = content.get('data')
        
        if content_type == 'text':
            return self.embed_text(data)
        elif content_type == 'image':
            image_format = content.get('format', 'png')
            return self.embed_image(data, image_format)
        else:
            logger.error(f"不支持的内容类型: {content_type}")
            return None

# 创建多模态嵌入器实例
try:
    multimodal_embedder = MultiModalEmbeddings()
    logger.info("多模态嵌入器初始化成功")
except Exception as e:
    multimodal_embedder = None
    logger.warning(f"多模态嵌入器初始化失败: {e}")

class AdaptiveEmbeddingRouter:
    """
    智能嵌入路由器：根据内容类型和长度自动选择最合适的嵌入策略
    
    策略：
    - 纯文本 ≤ 30K tokens: text-embedding-v4
    - 纯文本 > 30K tokens: 分块 + text-embedding-v4 + 平均向量
    - 多模态内容: 文本用text-embedding-v4 + 图片用multimodal-embedding-v1
    - 只有图片: multimodal-embedding-v1
    """
    
    def __init__(self):
        self.text_embedder = embedder_qwen  # text-embedding-v4, 8192 tokens
        self.multimodal_embedder = multimodal_embedder  # multimodal-embedding-v1, 512 tokens
        # 与 DashScope text-embedding-v4 的限制对齐（官方 8192 tokens）
        self.text_token_limit = 8192
        self.multimodal_text_limit = 500  # 为512留安全边际
    
    def detect_content_type(self, content: Dict) -> str:
        """
        检测内容类型
        
        Args:
            content: 包含text, images, videos等字段的内容字典
            
        Returns:
            content_type: 'text_only', 'image_only', 'multimodal', 'large_text'
        """
        has_text = bool(content.get('text', '').strip())
        has_images = bool(content.get('images', []))
        has_videos = bool(content.get('videos', []))
        
        text_length = 0
        if has_text:
            # 粗略估计token数量，中文约2-3字符一个token
            text_length = len(enc.encode(content['text']))
        
        if has_images or has_videos:
            if has_text:
                if text_length > self.text_token_limit:
                    return 'large_multimodal'
                else:
                    return 'multimodal'
            else:
                return 'media_only'
        elif has_text:
            if text_length > self.text_token_limit:
                return 'large_text'
            else:
                return 'text_only'
        else:
            return 'empty'
    
    def chunk_large_text(self, text: str, chunk_size: int = None, overlap: int = 1000) -> List[str]:
        """
        将超长文本智能分块
        优先在句子边界分割
        """
        # 依据当前文本嵌入模型限制动态确定块大小（为上限预留安全边际）
        if chunk_size is None:
            # DashScope text-embedding-v4 支持 8192 tokens，留出200 token缓冲
            chunk_size = max(1, self.text_token_limit - 200)
        
        if len(enc.encode(text)) <= chunk_size:
            return [text]
        
        import re
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(enc.encode(para))
            
            if current_tokens + para_tokens <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落就超限，进一步分割
                if para_tokens > chunk_size:
                    sentences = re.split(r'[。！？\.!?;；]\s*', para)
                    sub_chunk = ""
                    sub_tokens = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        sentence_tokens = len(enc.encode(sentence))
                        
                        if sub_tokens + sentence_tokens <= chunk_size:
                            sub_chunk += sentence + "。"
                            sub_tokens += sentence_tokens
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = sentence + "。"
                            sub_tokens = sentence_tokens
                    
                    if sub_chunk:
                        current_chunk = sub_chunk
                        current_tokens = sub_tokens
                    else:
                        current_chunk = ""
                        current_tokens = 0
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def embed_content(self, content: Dict) -> Dict:
        """
        根据内容类型智能选择嵌入策略
        
        Args:
            content: 包含text, images等字段的内容字典
            
        Returns:
            嵌入结果字典，包含text_embeddings, image_embeddings等
        """
        content_type = self.detect_content_type(content)
        result = {
            'content_type': content_type,
            'text_embeddings': [],
            'image_embeddings': [],
            'video_embeddings': [],
            'metadata': {}
        }
        
        try:
            if content_type == 'text_only':
                # 纯文本，根据长度决定是否分块
                token_len = len(enc.encode(content['text'] or ''))
                if token_len > self.text_token_limit:
                    chunks = self.chunk_large_text(content['text'])
                    embeddings = []
                    for chunk in chunks:
                        try:
                            if chunk and chunk.strip():
                                emb = self.text_embedder.embed_query(chunk)
                                if emb:
                                    embeddings.append(emb)
                        except Exception as e:
                            logger.warning(f"文本块嵌入失败: {e}")
                    result['text_embeddings'] = embeddings
                    result['metadata']['text_chunks'] = len(embeddings)
                    if embeddings:
                        import numpy as np
                        result['text_embedding_avg'] = list(np.mean(embeddings, axis=0))
                else:
                    text = (content['text'] or '').strip()
                    if text:
                        text_emb = self.text_embedder.embed_query(text)
                        result['text_embeddings'] = [text_emb]
                        result['metadata']['text_chunks'] = 1

            elif content_type == 'large_text':
                # 超长文本，严格按模型上限分块
                chunks = self.chunk_large_text(content['text'])
                embeddings = []
                
                for chunk in chunks:
                    try:
                        if chunk and chunk.strip():
                            emb = self.text_embedder.embed_query(chunk)
                        if emb:
                            embeddings.append(emb)
                    except Exception as e:
                        logger.warning(f"文本块嵌入失败: {e}")
                        continue
                
                result['text_embeddings'] = embeddings
                result['metadata']['text_chunks'] = len(embeddings)
                
                # 计算平均向量作为整体表示
                if embeddings:
                    import numpy as np
                    result['text_embedding_avg'] = list(np.mean(embeddings, axis=0))
                
            elif content_type in ['multimodal', 'large_multimodal']:
                # 多模态内容
                
                # 处理文本部分
                if content.get('text'):
                    if content_type == 'large_multimodal':
                        chunks = self.chunk_large_text(content['text'])
                        text_embeddings = []
                        for chunk in chunks:
                            try:
                                if chunk and chunk.strip():
                                    emb = self.text_embedder.embed_query(chunk)
                                if emb:
                                    text_embeddings.append(emb)
                            except Exception as e:
                                logger.warning(f"多模态文本块嵌入失败: {e}")
                        result['text_embeddings'] = text_embeddings
                        
                        if text_embeddings:
                            import numpy as np
                            result['text_embedding_avg'] = list(np.mean(text_embeddings, axis=0))
                    else:
                        text_emb = self.text_embedder.embed_query(content['text'])
                        result['text_embeddings'] = [text_emb]
                
                # 处理图片部分
                if content.get('images'):
                    image_embeddings = []
                    for image_data in content['images']:
                        try:
                            img_emb = self.multimodal_embedder.embed_image(
                                image_data.get('data'), 
                                image_data.get('format', 'png')
                            )
                            if img_emb:
                                image_embeddings.append(img_emb)
                        except Exception as e:
                            logger.warning(f"图片嵌入失败: {e}")
                    
                    result['image_embeddings'] = image_embeddings
                    result['metadata']['image_count'] = len(image_embeddings)
                
            elif content_type == 'media_only':
                # 只有图片/视频
                if content.get('images'):
                    image_embeddings = []
                    for image_data in content['images']:
                        try:
                            img_emb = self.multimodal_embedder.embed_image(
                                image_data.get('data'),
                                image_data.get('format', 'png')
                            )
                            if img_emb:
                                image_embeddings.append(img_emb)
                        except Exception as e:
                            logger.warning(f"纯图片嵌入失败: {e}")
                    
                    result['image_embeddings'] = image_embeddings
                    result['metadata']['image_count'] = len(image_embeddings)
            
        except Exception as e:
            logger.error(f"内容嵌入失败: {e}")
            result['error'] = str(e)
        
        return result

# MultiModalEmbeddingAdapter类定义
class MultiModalEmbeddingAdapter:
    """
    LangChain兼容的多模态embedding适配器
    将MultiModalEmbeddings接口适配为LangChain标准接口
    """
    
    def __init__(self, multimodal_embedder):
        self.multimodal_embedder = multimodal_embedder
    
    def embed_query(self, text: str) -> List[float]:
        """LangChain标准接口：嵌入单个查询"""
        return self.multimodal_embedder.embed_text(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain标准接口：嵌入多个文档"""
        return self.multimodal_embedder.embed_texts(texts)

# 创建智能路由器实例
try:
    adaptive_embedder = AdaptiveEmbeddingRouter()
    # 创建适配器
    multimodal_embedder_adapter = MultiModalEmbeddingAdapter(multimodal_embedder) if multimodal_embedder else None
    logger.info("智能嵌入路由器初始化成功")
except Exception as e:
    adaptive_embedder = None
    multimodal_embedder_adapter = None
    logger.warning(f"智能嵌入路由器初始化失败: {e}")

# 为了向后兼容，保留原有的embedder_qwen

# # BGE embedding model
# client = openai.Client(
#     base_url="http://172.31.1.109:13000/v1",
#     api_key="sk-zE5I0zw01FA89aWX97D6F52e803a4124B13c54095dF3A7Db"
# )
# # Chat completion
# def embedding_vector_openai(input):
#     response = client.embeddings.create(model="bge-m3:latest",
#                                           input= [input],
#                                           )
#     res = [item.embedding for item in response.data][0]
#     return res

#%%
"""
Embedder for BGE-M3 using FlagEmbedding.
"""
# import numpy as np
# from langchain.embeddings.base import Embeddings
# from FlagEmbedding import BGEM3FlagModel
#
# BGE_M3 = BGEM3FlagModel("/Users/depblu/Documents/GitHub/genQA/models/BAAI/bge-m3", use_fp16=False, device="cuda")
#
# class BGEM3Embeddings(Embeddings):
#     def __init__(self, model=BGE_M3):
#         self.model = model            # BGE_M3 构造时改 use_fp16=False
#
#     def _to_fp32_list(self, arr):
#         import numpy as np
#         return (arr.astype(np.float32) if isinstance(arr, np.ndarray) else
#                 np.asarray(arr, dtype=np.float32)).tolist()
#
#     def embed_documents(self, texts):
#         if not texts:
#             return []
#         vecs = self.model.encode(texts, max_length=1024)["dense_vecs"]
#         return self._to_fp32_list(vecs)
#
#     def embed_query(self, text):
#         return self.embed_documents([text])[0]
