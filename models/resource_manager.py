"""
资源管理器模块
负责统一管理模型、媒体资源和配置信息
"""

import os
import json
import pickle
import threading
from typing import Dict, Any, List, Optional, Union
import traceback

class ResourceManager:
    """资源管理器，负责模型、媒体资源和配置的统一管理"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化资源管理器"""
        if self._initialized:
            return
            
        # 获取项目根目录
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 资源目录
        self.models_dir = os.path.join(self.base_path, 'data', 'models')
        self.training_dir = os.path.join(self.base_path, 'data', 'training')
        self.media_dir = os.path.join(self.base_path, 'data', 'media')
        self.generated_dir = os.path.join(self.base_path, 'data', 'generated')
        self.config_path = os.path.join(self.base_path, 'data', 'config.json')
        
        # 确保所有目录存在
        for path in [self.models_dir, self.training_dir, self.media_dir, 
                     os.path.join(self.generated_dir, 'images'),
                     os.path.join(self.generated_dir, 'videos')]:
            os.makedirs(path, exist_ok=True)
        
        # 模型资源
        self.models = {}  # 模型ID -> 模型实例
        self.current_model_id = None
        
        # 媒体资源映射
        self.media_resources = {}  # tag -> {type: "image/video", path: "file_path"}
        
        # 配置信息
        self.config = self._load_config()
        
        # 资源锁
        self.model_locks = {}  # 模型ID -> 锁
        self.global_locks = {
            "image_generator": threading.RLock(),
            "video_generator": threading.RLock(),
            "training": threading.RLock()
        }
        
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置信息"""
        default_config = {
            "confidence_threshold": 0.3,
            "debug_mode": False,
            "default_model": "chat_model.pth",
            "ui": {
                "theme": "light",
                "font_family": "Microsoft YaHei",
                "font_size": 10
            },
            "advanced": {
                "transformer_model": "THUDM/chatglm3-6b",
                "stable_diffusion_model": "runwayml/stable-diffusion-v1-5"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置和加载的配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict) and isinstance(config[key], dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
            except Exception as e:
                print(f"加载配置文件时出错: {str(e)}")
                return default_config
        else:
            # 保存默认配置
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存默认配置时出错: {str(e)}")
            return default_config
    
    def save_config(self) -> bool:
        """保存配置信息"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置时出错: {str(e)}")
            return False
    
    def get_model_path(self, model_id: Optional[str] = None) -> str:
        """获取模型路径"""
        if model_id is None:
            model_id = self.config.get("default_model", "chat_model.pth")
        
        if os.path.isabs(model_id):
            return model_id
        else:
            return os.path.join(self.models_dir, model_id)
    
    def load_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """加载模型
        
        Args:
            model_id: 模型ID，如果为None则加载默认模型
            
        Returns:
            Dict[str, Any]: 加载结果信息
        """
        result = {
            "success": False,
            "model": None,
            "error": None,
            "model_id": model_id
        }
        
        try:
            model_path = self.get_model_path(model_id)
            
            if not os.path.exists(model_path):
                result["error"] = f"模型文件不存在: {model_path}"
                return result
            
            # 加载模型
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 更新当前模型ID
            self.current_model_id = model_id
            
            # 将模型数据存储在模型字典中
            self.models[model_id] = model_data
            
            # 更新媒体资源
            if "media_resources" in model_data:
                self.media_resources.update(model_data["media_resources"])
            
            result["success"] = True
            result["model"] = model_data
            
            return result
        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()
            return result
    
    def get_lock(self, resource_id: str) -> threading.RLock:
        """获取资源锁
        
        Args:
            resource_id: 资源ID，可以是模型ID或全局资源ID
            
        Returns:
            threading.RLock: 资源锁
        """
        if resource_id in self.global_locks:
            return self.global_locks[resource_id]
        
        if resource_id not in self.model_locks:
            self.model_locks[resource_id] = threading.RLock()
        
        return self.model_locks[resource_id]
    
    def get_media_resource(self, tag: str) -> Optional[Dict[str, str]]:
        """获取媒体资源
        
        Args:
            tag: 标签
            
        Returns:
            Optional[Dict[str, str]]: 媒体资源信息，如果不存在则返回None
        """
        return self.media_resources.get(tag)
    
    def register_media_resource(self, tag: str, media_type: str, media_path: str) -> bool:
        """注册媒体资源
        
        Args:
            tag: 标签
            media_type: 媒体类型(image/video)
            media_path: 媒体路径
            
        Returns:
            bool: 是否注册成功
        """
        if not os.path.exists(media_path):
            return False
        
        self.media_resources[tag] = {
            "type": media_type,
            "path": media_path
        }
        return True
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.config.get("ui", {})
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """获取高级配置"""
        return self.config.get("advanced", {})
    
    def get_confidence_threshold(self) -> float:
        """获取置信度阈值"""
        return self.config.get("confidence_threshold", 0.3)
    
    def set_confidence_threshold(self, threshold: float) -> bool:
        """设置置信度阈值"""
        if 0 <= threshold <= 1:
            self.config["confidence_threshold"] = threshold
            return self.save_config()
        return False
    
    def is_debug_mode(self) -> bool:
        """获取调试模式状态"""
        return self.config.get("debug_mode", False)
    
    def set_debug_mode(self, enabled: bool) -> bool:
        """设置调试模式状态"""
        self.config["debug_mode"] = enabled
        return self.save_config()
    
    def get_generated_path(self, file_type: str, filename: str) -> str:
        """获取生成文件的保存路径
        
        Args:
            file_type: 文件类型(images/videos)
            filename: 文件名
            
        Returns:
            str: 完整文件路径
        """
        return os.path.join(self.generated_dir, file_type, filename)
    
    def cleanup(self):
        """清理资源"""
        # 保存配置
        self.save_config()
        
        # 清理模型
        for model_id in list(self.models.keys()):
            self.models[model_id] = None
        
        self.models.clear()
        self.media_resources.clear() 