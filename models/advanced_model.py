import os
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging
import time
import random
import json
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedModel")

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch库未安装，高级模型功能将不可用")
    TORCH_AVAILABLE = False

# 尝试导入PIL
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL库未安装，图像处理功能将不可用")
    PIL_AVAILABLE = False

# 尝试导入TextBlob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    logger.warning("TextBlob库未安装，情感分析功能将不可用")
    TEXTBLOB_AVAILABLE = False

# 尝试导入TensorFlow
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow库导入失败: {str(e)}")
    logger.warning("依赖TensorFlow的功能将不可用")

# 尝试导入Transformers
TRANSFORMERS_AVAILABLE = False
try:
    # 在导入transformers前设置环境变量，告诉transformers不要自动导入tensorflow
    # 这可以防止因为tensorflow导入失败而导致transformers完全不可用
    import os
    os.environ["USE_TORCH"] = "1"
    os.environ["USE_TF"] = "0"
    
    # 尝试导入transformers的子模块，分别处理可能的错误
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    
    # 尝试导入更多transformers组件，但不应该因为这些组件导入失败而阻止程序运行
    try:
        from transformers import (
            AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
            pipeline, BertTokenizer, BertModel
        )
        logger.info("Transformers完整功能可用")
    except ImportError as e:
        logger.warning(f"部分Transformers功能不可用: {str(e)}")
except ImportError as e:
    logger.warning(f"Transformers库导入失败: {str(e)}")
    logger.warning("高级语言模型功能将不可用")

# 尝试导入Diffusers
DIFFUSERS_AVAILABLE = False
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("Diffusers库未安装，图像生成功能将不可用")

# 尝试导入SentenceTransformers
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers库未安装，句子嵌入功能将不可用")

# 尝试导入MoviePy
MOVIEPY_AVAILABLE = False
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    logger.warning("MoviePy库未安装，视频处理功能将不可用")

class EmotionAnalyzer:
    """情感分析器"""
    def __init__(self):
        self.initialized = TEXTBLOB_AVAILABLE
    
    def analyze(self, text: str) -> Dict[str, float]:
        """分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict[str, float]: 情感分析结果，包含极性和主观性
        """
        if not self.initialized:
            return {"polarity": 0.0, "subjectivity": 0.0}
        
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # 极性范围：[-1.0, 1.0]，-1表示非常负面，1表示非常正面
        # 主观性范围：[0.0, 1.0]，0表示非常客观，1表示非常主观
        return {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }

class ImageGenerator:
    """图像生成器"""
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.initialized = False
    
    def initialize(self):
        """初始化图像生成模型"""
        if self.initialized:
            return True
            
        if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("无法初始化图像生成模型：缺少必要的库")
            return False
            
        try:
            logger.info(f"正在初始化图像生成模型: {self.model_id}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)
            self.initialized = True
            logger.info("图像生成模型初始化完成")
            return True
        except Exception as e:
            logger.error(f"初始化图像生成模型失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate(self, prompt: str, negative_prompt: str = None, width: int = 512, height: int = 512) -> Optional[Image.Image]:
        """生成图像
        
        Args:
            prompt: 生成提示文本
            negative_prompt: 负面提示，可选
            width: 图像宽度
            height: 图像高度
            
        Returns:
            PIL.Image.Image or None: 生成的图像
        """
        if not DIFFUSERS_AVAILABLE or not PIL_AVAILABLE:
            logger.error("图像生成功能不可用：缺少必要的库")
            return None
            
        if not self.initialized and not self.initialize():
            return None
            
        try:
            params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }
            
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
                
            logger.info(f"开始生成图像，提示词: {prompt}")
            start_time = time.time()
            
            result = self.pipe(**params)
            image = result.images[0]
            
            elapsed = time.time() - start_time
            logger.info(f"图像生成完成，耗时: {elapsed:.2f}秒")
            
            return image
        except Exception as e:
            logger.error(f"生成图像时出错: {str(e)}")
            traceback.print_exc()
            return None
    
    def add_watermark(self, image: Image.Image, text: str = "AI生成") -> Image.Image:
        """添加水印
        
        Args:
            image: 原始图像
            text: 水印文本
            
        Returns:
            PIL.Image.Image: 添加水印后的图像
        """
        try:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("simhei.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # 在右下角添加水印
            text_width, text_height = draw.textsize(text, font=font)
            x = width - text_width - 10
            y = height - text_height - 10
            
            # 首先绘制黑色阴影
            draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0, 128))
            # 然后绘制白色文本
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))
            
            return image
        except Exception as e:
            logger.warning(f"添加水印时出错: {str(e)}")
            return image

class VideoGenerator:
    """视频生成器"""
    def __init__(self):
        self.image_generator = ImageGenerator()
        self.initialized = MOVIEPY_AVAILABLE and PIL_AVAILABLE
        
    def generate_from_frames(self, frames: List[Image.Image], output_path: str, fps: int = 24) -> str:
        """从图像帧生成视频
        
        Args:
            frames: 图像帧列表
            output_path: 输出视频路径
            fps: 帧率
            
        Returns:
            str: 视频文件路径
        """
        if not self.initialized:
            logger.error("视频生成功能不可用：缺少必要的库")
            return ""
            
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 创建临时图像文件夹
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有帧为图像
            frame_files = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                frame.save(frame_path)
                frame_files.append(frame_path)
            
            # 使用MoviePy生成视频
            clips = [mp.ImageClip(f).set_duration(1/fps) for f in frame_files]
            concat_clip = mp.concatenate_videoclips(clips, method="compose")
            concat_clip.write_videofile(output_path, fps=fps)
            
            # 清理临时文件
            for f in frame_files:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                
            return output_path
        except Exception as e:
            logger.error(f"生成视频时出错: {str(e)}")
            traceback.print_exc()
            return ""
    
    def generate_from_prompt(self, prompt: str, frame_count: int = 10, output_path: str = None) -> str:
        """根据提示词生成视频
        
        Args:
            prompt: 提示词
            frame_count: 帧数量
            output_path: 输出路径，可选
            
        Returns:
            str: 视频文件路径
        """
        if not self.initialized:
            logger.error("视频生成功能不可用：缺少必要的库")
            return ""
            
        if not output_path:
            # 生成默认输出路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.expanduser("~"), "generated_videos")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
        
        try:
            logger.info(f"开始生成视频，提示词: {prompt}，帧数: {frame_count}")
            start_time = time.time()
            
            # 生成一系列略有变化的图像
            frames = []
            for i in range(frame_count):
                # 略微变化提示词，使生成的帧有连续性
                frame_prompt = f"{prompt}, frame {i+1} of sequence"
                
                # 生成图像
                image = self.image_generator.generate(frame_prompt)
                if image:
                    frames.append(image)
                    logger.info(f"已生成第 {i+1}/{frame_count} 帧")
                else:
                    logger.warning(f"第 {i+1}/{frame_count} 帧生成失败")
            
            if not frames:
                logger.error("没有生成任何有效帧，视频生成失败")
                return ""
                
            # 从帧生成视频
            video_path = self.generate_from_frames(frames, output_path, fps=24)
            
            elapsed = time.time() - start_time
            logger.info(f"视频生成完成，耗时: {elapsed:.2f}秒，保存到: {video_path}")
            
            return video_path
            
        except Exception as e:
            logger.error(f"根据提示词生成视频时出错: {str(e)}")
            traceback.print_exc()
            return ""

class DialogContext:
    """对话上下文管理器"""
    
    def __init__(self, max_history: int = 10):
        """初始化
        
        Args:
            max_history: 最大历史记录数量
        """
        self.history = []
        self.max_history = max_history
        self.metadata = {}
        
    def add(self, role: str, content: str, **kwargs) -> None:
        """添加一条消息到历史记录
        
        Args:
            role: 角色（"user"或"assistant"）
            content: 消息内容
            **kwargs: 其他元数据
        """
        # 创建消息记录
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        # 添加到历史记录
        self.history.append(message)
        
        # 如果超过最大历史记录数量，移除最早的记录
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整历史记录"""
        return self.history
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """获取格式化的历史记录，适用于OpenAI API等"""
        formatted = []
        for msg in self.history:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted
    
    def get_as_text(self, include_roles: bool = True) -> str:
        """将历史记录转换为文本格式
        
        Args:
            include_roles: 是否包含角色前缀
            
        Returns:
            str: 文本格式的历史记录
        """
        text = ""
        for msg in self.history:
            if include_roles:
                prefix = f"{msg['role'].title()}: "
            else:
                prefix = ""
            text += f"{prefix}{msg['content']}\n\n"
        return text.strip()
    
    def clear(self) -> None:
        """清空历史记录"""
        self.history = []
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "history": self.history,
            "metadata": self.metadata,
            "max_history": self.max_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogContext':
        """从字典创建对话上下文"""
        context = cls(max_history=data.get("max_history", 10))
        context.history = data.get("history", [])
        context.metadata = data.get("metadata", {})
        return context

class TransformerModel:
    """Transformer模型封装"""
    
    def __init__(
        self, 
        model_name_or_path: str = "THUDM/chatglm3-6b", 
        model_type: str = "causal",
        device: str = None
    ):
        """初始化
        
        Args:
            model_name_or_path: 模型名称或路径
            model_type: 模型类型，"causal"或"seq2seq"
            device: 设备，"cuda"或"cpu"
        """
        self.model_name = model_name_or_path
        self.model_type = model_type
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.context = DialogContext()
        
    def initialize(self) -> bool:
        """初始化模型"""
        if self.initialized:
            return True
            
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.error(f"无法初始化Transformer模型：缺少必要的库")
            return False
            
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            start_time = time.time()
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 根据类型加载模型
            if self.model_type.lower() == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
            # 将模型移动到设备
            self.model = self.model.to(self.device)
            
            elapsed = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {elapsed:.2f}秒")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"初始化模型时出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate(
        self, 
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = None
    ) -> str:
        """生成文本
        
        Args:
            prompt: 提示词
            max_length: 最大生成长度
            temperature: 温度系数
            top_p: 核采样阈值
            system_prompt: 系统提示词
            
        Returns:
            str: 生成的文本
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            return f"无法生成回复：缺少必要的库。您的问题是：{prompt}"
            
        if not self.initialized and not self.initialize():
            return f"无法生成回复：模型初始化失败。您的问题是：{prompt}"
            
        try:
            # 添加系统提示和历史记录
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
                
            # 添加历史记录
            if self.context.history:
                history_text = self.context.get_as_text()
                full_prompt += f"{history_text}\n\n"
                
            # 添加当前提示词
            full_prompt += f"User: {prompt}\nAssistant: "
            
            # 记录当前提示词到上下文
            self.context.add("user", prompt)
            
            # 生成回复
            logger.info(f"开始生成回复，提示词长度: {len(full_prompt)}")
            start_time = time.time()
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回复部分
            if "Assistant: " in response:
                response = response.split("Assistant: ")[-1].strip()
                
            elapsed = time.time() - start_time
            logger.info(f"回复生成完成，耗时: {elapsed:.2f}秒，长度: {len(response)}")
            
            # 记录回复到上下文
            self.context.add("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"生成回复时出错: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            return f"生成回复时出错: {str(e)}"
    
    def clear_context(self) -> None:
        """清空对话上下文"""
        self.context.clear()

class MultiModalModel:
    """多模态模型"""
    
    def __init__(self, transformer_model_name: str = "THUDM/chatglm3-6b"):
        # 初始化状态跟踪
        self.initialized = False
        self.initialization_errors = {}
        self.components_available = {
            "transformer": False,
            "image_generator": False,
            "video_generator": False,
            "emotion_analyzer": False,
            "sentence_embedding": False
        }
        
        # 初始化组件
        self.transformer_model_name = transformer_model_name
        self.transformer = None
        self.image_generator = None
        self.video_generator = None
        self.emotion_analyzer = None
        self.sentence_model = None
        
        # 情感分析器
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 对话上下文
        self.context = DialogContext(max_history=15)
    
    def initialize(self) -> Dict[str, Any]:
        """初始化所有模型
        
        Returns:
            Dict[str, Any]: 包含初始化状态和错误信息的字典
        """
        init_results = {}
        
        # 初始化文本模型
        if self.transformer_model_name:
            try:
                self.transformer = TransformerModel(model_name_or_path=self.transformer_model_name)
                self.components_available["transformer"] = True
                init_results["transformer"] = {
                    "success": True,
                    "error": None
                }
                logger.info("文本模型已创建")
            except Exception as e:
                error_msg = f"文本模型初始化失败: {str(e)}"
                init_results["transformer"] = {"success": False, "error": error_msg}
                self.initialization_errors["transformer"] = error_msg
                logger.error(error_msg)
        else:
            init_results["transformer"] = {"success": False, "error": "文本模型不可用"}
        
        # 尝试初始化图像生成器
        try:
            self.image_generator = ImageGenerator()
            self.components_available["image_generator"] = True
            init_results["image_generator"] = {
                "success": True,
                "error": None
            }
            logger.info("图像生成器已创建")
        except Exception as e:
            self.image_generator = None
            error_msg = f"图像生成器初始化失败: {str(e)}"
            init_results["image_generator"] = {"success": False, "error": error_msg}
            self.initialization_errors["image_generator"] = error_msg
            logger.error(error_msg)
        
        # 尝试初始化视频生成器
        try:
            self.video_generator = VideoGenerator() if self.image_generator else None
            if self.video_generator:
                self.components_available["video_generator"] = True
                init_results["video_generator"] = {
                    "success": True,
                    "error": None
                }
                logger.info("视频生成器已创建")
            else:
                logger.warning("由于图像生成器不可用，视频生成器也无法创建")
                self.initialization_errors["video_generator"] = "依赖的图像生成器不可用"
        except Exception as e:
            self.video_generator = None
            error_msg = f"视频生成器初始化失败: {str(e)}"
            init_results["video_generator"] = {"success": False, "error": error_msg}
            self.initialization_errors["video_generator"] = error_msg
            logger.error(error_msg)
        
        # 图片和视频存储目录
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_dir = os.path.join(self.base_path, "data", "generated", "images")
        self.video_dir = os.path.join(self.base_path, "data", "generated", "videos")
        
        # 创建必要的目录
        try:
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.video_dir, exist_ok=True)
            logger.info("已创建生成内容存储目录")
        except Exception as e:
            error_msg = f"创建目录失败: {str(e)}"
            self.initialization_errors["storage"] = error_msg
            logger.error(error_msg)
        
        self.initialized = all(self.components_available.values())
        return init_results
    
    def get_status(self) -> Dict[str, bool]:
        """获取各模块可用性状态
        
        Returns:
            Dict[str, bool]: 模块可用性状态字典
        """
        return self.components_available.copy()
    
    def get_initialization_errors(self) -> Dict[str, str]:
        """获取初始化过程中的错误信息
        
        Returns:
            Dict[str, str]: 错误信息字典
        """
        return self.initialization_errors.copy()
        
    def process_request(self, text: str) -> Dict[str, Any]:
        """处理用户请求，根据内容生成不同类型的回复
        
        Args:
            text: 用户输入
            
        Returns:
            Dict[str, Any]: 包含不同类型输出的字典
        """
        result = {
            "text": None,
            "image_path": None,
            "video_path": None,
            "type": "text",  # 默认类型
            "success": True,
            "error": None
        }
        
        # 检查是否包含图像生成请求
        if any(keyword in text.lower() for keyword in ["生成图片", "创建图像", "画", "绘制", "生成一张", "绘制一张"]):
            # 检查图像生成功能是否可用
            if not self.components_available["image_generator"]:
                error_msg = f"图像生成功能不可用: {self.initialization_errors.get('image_generator', '未知错误')}"
                result["text"] = f"抱歉，我无法生成图片。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                return result
                
            # 提取图像描述
            prompt = text.split("生成图片", 1)[-1] if "生成图片" in text else text
            prompt = prompt.split("创建图像", 1)[-1] if "创建图像" in prompt else prompt
            prompt = prompt.split("画", 1)[-1] if "画" in prompt else prompt
            prompt = prompt.split("绘制", 1)[-1] if "绘制" in prompt else prompt
            prompt = prompt.split("生成一张", 1)[-1] if "生成一张" in prompt else prompt
            prompt = prompt.split("绘制一张", 1)[-1] if "绘制一张" in prompt else prompt
            
            # 生成图像
            try:
                logger.info(f"开始生成图像，提示词: {prompt}")
                image = self.image_generator.generate(prompt)
                
                if image:
                    # 添加水印
                    image = self.image_generator.add_watermark(image)
                    
                    # 保存图像
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(self.image_dir, f"image_{timestamp}.png")
                    try:
                        image.save(image_path)
                        result["image_path"] = image_path
                        result["type"] = "image"
                        result["text"] = f"我已根据您的描述生成了一张图片，图片路径: {image_path}"
                        logger.info(f"图像生成成功，保存到: {image_path}")
                    except Exception as e:
                        error_msg = f"保存图像失败: {str(e)}"
                        result["text"] = f"图像生成成功，但保存失败。{error_msg}"
                        result["success"] = False
                        result["error"] = error_msg
                        logger.error(error_msg)
                else:
                    error_msg = "图像生成返回空结果，可能是提示词不适合或格式不正确"
                    result["text"] = f"很抱歉，图像生成失败。{error_msg}"
                    result["success"] = False
                    result["error"] = error_msg
                    logger.error(error_msg)
            except Exception as e:
                error_msg = f"图像生成过程出错: {str(e)}"
                result["text"] = f"很抱歉，图像生成失败。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
        
        # 检查是否包含视频生成请求
        elif any(keyword in text.lower() for keyword in ["生成视频", "创建视频", "录制", "动画", "生成一段视频"]):
            # 检查视频生成功能是否可用
            if not self.components_available["video_generator"]:
                error_msg = f"视频生成功能不可用: {self.initialization_errors.get('video_generator', '未知错误')}"
                result["text"] = f"抱歉，我无法生成视频。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                return result
                
            # 提取视频描述
            prompt = text.split("生成视频", 1)[-1] if "生成视频" in text else text
            prompt = prompt.split("创建视频", 1)[-1] if "创建视频" in prompt else prompt
            prompt = prompt.split("录制", 1)[-1] if "录制" in prompt else prompt
            prompt = prompt.split("动画", 1)[-1] if "动画" in prompt else prompt
            prompt = prompt.split("生成一段视频", 1)[-1] if "生成一段视频" in prompt else prompt
            
            # 生成视频
            try:
                logger.info(f"开始生成视频，提示词: {prompt}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(self.video_dir, f"video_{timestamp}.mp4")
                
                video_path = self.video_generator.generate_from_prompt(
                    prompt=prompt,
                    frame_count=5,  # 减少帧数加快生成
                    output_path=video_path
                )
                
                if video_path and os.path.exists(video_path):
                    result["video_path"] = video_path
                    result["type"] = "video"
                    result["text"] = f"我已根据您的描述生成了一段视频，视频路径: {video_path}"
                    logger.info(f"视频生成成功，保存到: {video_path}")
                else:
                    error_msg = "视频生成返回空结果或文件不存在，可能是提示词不适合或格式不正确"
                    result["text"] = f"很抱歉，视频生成失败。{error_msg}"
                    result["success"] = False
                    result["error"] = error_msg
                    logger.error(error_msg)
            except Exception as e:
                error_msg = f"视频生成过程出错: {str(e)}"
                result["text"] = f"很抱歉，视频生成失败。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
        
        # 普通文本回复
        else:
            # 检查文本生成功能是否可用
            if not self.components_available["transformer"]:
                error_msg = f"文本生成功能不可用: {self.initialization_errors.get('transformer', '未知错误')}"
                result["text"] = f"抱歉，我无法生成回复。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                return result
                
            try:
                response = self.transformer.generate(text)
                result["text"] = response
                logger.info(f"文本回复生成成功，长度: {len(response)}")
            except Exception as e:
                error_msg = f"文本生成过程出错: {str(e)}"
                result["text"] = f"很抱歉，生成回复时出错。{error_msg}"
                result["success"] = False
                result["error"] = error_msg
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
        
        return result
    
    def clear_context(self) -> None:
        """清空对话上下文"""
        if self.components_available["transformer"] and self.transformer:
            try:
                self.transformer.clear_context()
                logger.info("对话上下文已清空")
            except Exception as e:
                logger.error(f"清空对话上下文时出错: {str(e)}")
                logger.debug(traceback.format_exc()) 