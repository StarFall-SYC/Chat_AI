"""
实用工具模块
提供各种通用的辅助函数
"""

import os
import sys
import uuid
import time
import random
import string
import hashlib
import datetime
from typing import List, Dict, Any, Optional, Tuple

from .logger import app_logger


def setup_environment():
    """设置应用程序运行环境
    
    确保必要的目录结构存在，初始化环境变量等
    
    Returns:
        bool: 是否设置成功
    """
    try:
        # 获取项目根目录
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 要创建的目录列表
        directories = [
            os.path.join(base_path, 'data', 'models'),               # 模型保存目录
            os.path.join(base_path, 'data', 'training'),             # 训练数据目录
            os.path.join(base_path, 'data', 'history'),              # 历史记录目录
            os.path.join(base_path, 'data', 'logs'),                 # 日志目录
            os.path.join(base_path, 'data', 'generated', 'images'),  # 生成图像目录
            os.path.join(base_path, 'data', 'generated', 'videos'),  # 生成视频目录
        ]
        
        # 创建目录
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                app_logger.debug(f"已创建目录: {directory}")
        
        # 设置其他环境变量等
        
        return True
    except Exception as e:
        app_logger.error(f"环境设置失败: {str(e)}")
        return False


def generate_unique_id(prefix: str = "id") -> str:
    """生成唯一ID
    
    Args:
        prefix: ID前缀
        
    Returns:
        str: 唯一ID
    """
    unique_id = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{prefix}_{unique_id}"


def generate_timestamp() -> str:
    """生成时间戳字符串
    
    Returns:
        str: 时间戳字符串，格式：YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: MD5哈希值
    """
    if not os.path.exists(file_path):
        return ""
    
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # 64KB chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
    except Exception as e:
        app_logger.error(f"计算文件哈希值失败: {str(e)}")
        return ""
    
    return hasher.hexdigest()


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 清理后的文件名
    """
    # 替换Windows文件系统中的非法字符
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # 确保文件名不超过255个字符
    if len(filename) > 255:
        base, ext = os.path.splitext(filename)
        filename = base[:255-len(ext)] + ext
    
    return filename


def time_execution(func):
    """计时装饰器，用于测量函数执行时间
    
    Args:
        func: 要测量的函数
        
    Returns:
        function: 装饰后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        app_logger.debug(f"{func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper


def ensure_dir(directory: str) -> bool:
    """确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: 是否成功创建或已存在
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        app_logger.error(f"创建目录 {directory} 失败: {str(e)}")
        return False


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件扩展名（小写，不包含点）
    """
    return os.path.splitext(file_path)[1].lower().lstrip('.')


def random_string(length: int = 8) -> str:
    """生成随机字符串
    
    Args:
        length: 字符串长度
        
    Returns:
        str: 随机字符串
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def format_size(size_bytes: int) -> str:
    """格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        str: 格式化后的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径
    
    在打包成可执行文件后，资源文件的路径可能会发生变化
    此函数用于确保能够正确获取资源文件路径
    
    Args:
        relative_path: 相对资源路径
        
    Returns:
        str: 资源文件的绝对路径
    """
    try:
        # 判断应用是否已打包
        if getattr(sys, 'frozen', False):
            # 获取应用所在目录
            application_path = os.path.dirname(sys.executable)
        else:
            # 获取脚本所在目录
            application_path = os.path.dirname(os.path.abspath(__file__))
            # 返回上一级目录
            application_path = os.path.dirname(application_path)
            
        path = os.path.join(application_path, relative_path)
        return path
    except Exception as e:
        app_logger.error(f"获取资源路径失败: {str(e)}")
        return relative_path


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本，超出长度的部分用省略号代替
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后的后缀
        
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix 