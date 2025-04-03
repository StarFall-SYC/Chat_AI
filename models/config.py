"""
配置管理模块
集中管理应用程序的各种配置项
"""

import os
import json
import threading
from typing import Dict, Any, Optional


class Config:
    """配置管理类，集中管理应用设置"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if getattr(self, "_initialized", False):
            return
            
        # 获取项目根目录
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 配置文件路径
        self.config_file = os.path.join(self.base_path, 'data', 'config.json')
        
        # 资源目录
        self.dirs = {
            "models": os.path.join(self.base_path, 'data', 'models'),
            "training": os.path.join(self.base_path, 'data', 'training'),
            "history": os.path.join(self.base_path, 'data', 'history'),
            "images": os.path.join(self.base_path, 'data', 'generated', 'images'),
            "videos": os.path.join(self.base_path, 'data', 'generated', 'videos'),
            "logs": os.path.join(self.base_path, 'data', 'logs'),
        }
        
        # 确保目录存在
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)
        
        # 默认配置
        self.defaults = {
            "app": {
                "version": "2.0.0",
                "name": "AI聊天大模型",
                "organization": "AI技术团队"
            },
            "ui": {
                "theme": "light",
                "font_family": "Microsoft YaHei",
                "font_size": 10
            },
            "model": {
                "default_model": "chat_model.pth",
                "confidence_threshold": 0.3,
                "transformer_model": "THUDM/chatglm3-6b",
                "stable_diffusion_model": "runwayml/stable-diffusion-v1-5",
                "max_history": 10,
                "debug_mode": False
            },
            "advanced": {
                "image_generation": {
                    "width": 512,
                    "height": 512,
                    "watermark": True
                },
                "video_generation": {
                    "frames": 10,
                    "fps": 24
                }
            }
        }
        
        # 加载配置
        self.settings = self._load_config()
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict[str, Any]: 配置项
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 合并默认配置，确保新增配置也存在
                merged = self._merge_configs(self.defaults, config)
                return merged
            except Exception as e:
                print(f"加载配置文件时出错: {str(e)}")
                return self.defaults.copy()
        else:
            # 保存默认配置
            self.save_config()
            return self.defaults.copy()
    
    def _merge_configs(self, defaults: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典
        
        Args:
            defaults: 默认配置
            config: 用户配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        result = defaults.copy()
        
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def save_config(self) -> bool:
        """保存配置到文件
        
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            section: 配置区域
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        if section in self.settings and key in self.settings[section]:
            return self.settings[section][key]
        return default
    
    def set(self, section: str, key: str, value: Any) -> bool:
        """设置配置项
        
        Args:
            section: 配置区域
            key: 配置键
            value: 配置值
            
        Returns:
            bool: 是否设置成功
        """
        if section not in self.settings:
            self.settings[section] = {}
            
        self.settings[section][key] = value
        return self.save_config()
    
    def get_dir(self, name: str) -> str:
        """获取目录路径
        
        Args:
            name: 目录名称
            
        Returns:
            str: 目录路径
        """
        return self.dirs.get(name, self.base_path)
    
    def get_app_info(self) -> Dict[str, str]:
        """获取应用信息
        
        Returns:
            Dict[str, str]: 应用信息
        """
        return self.settings.get("app", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置
        
        Returns:
            Dict[str, Any]: UI配置
        """
        return self.settings.get("ui", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置
        
        Returns:
            Dict[str, Any]: 模型配置
        """
        return self.settings.get("model", {})
        
    def get_advanced_config(self) -> Dict[str, Any]:
        """获取高级配置
        
        Returns:
            Dict[str, Any]: 高级配置
        """
        return self.settings.get("advanced", {})


# 导出单例实例
config = Config() 