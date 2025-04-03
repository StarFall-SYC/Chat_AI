"""
日志模块
提供统一的日志记录功能
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

from .config import config


class Logger:
    """应用程序日志记录器"""
    
    def __init__(self, name="ai_chat_model"):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 确保日志目录存在
        self.log_dir = config.get_dir("logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 日志文件路径
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(self.log_dir, f"{date_str}_{name}.log")
        
        # 创建日志处理器
        self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 清除已有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self._get_formatter(colored=True))
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(file_handler)
    
    def _get_formatter(self, colored=False):
        """获取日志格式器
        
        Args:
            colored: 是否使用彩色日志
            
        Returns:
            logging.Formatter: 日志格式器
        """
        if colored:
            # 控制台彩色输出
            format_str = (
                "\033[38;5;242m%(asctime)s\033[0m "  # 灰色时间戳
                "[\033[1;%(color)sm%(levelname)s\033[0m] "  # 彩色日志级别
                "\033[38;5;242m%(name)s:\033[0m "  # 灰色日志名称
                "%(message)s"  # 日志消息
            )
            return ColoredFormatter(format_str)
        else:
            # 文件输出
            format_str = (
                "%(asctime)s [%(levelname)s] %(name)s: "
                "%(message)s (%(filename)s:%(lineno)d)"
            )
            return logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    def debug(self, msg, *args, **kwargs):
        """记录调试日志"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """记录信息日志"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """记录警告日志"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """记录错误日志"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """记录异常日志"""
        self.logger.exception(msg, *args, **kwargs)


class ColoredFormatter(logging.Formatter):
    """彩色日志格式器"""
    
    # 日志级别对应的颜色代码
    COLORS = {
        'DEBUG': '244',     # 淡灰色
        'INFO': '39',       # 青色
        'WARNING': '208',   # 橙色
        'ERROR': '196',     # 红色
        'CRITICAL': '201',  # 粉色
    }
    
    def format(self, record):
        """格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            str: 格式化后的日志
        """
        record.color = self.COLORS.get(record.levelname, '37')
        return super().format(record)


# 创建应用程序级别的日志记录器
app_logger = Logger("app")
model_logger = Logger("model")
ui_logger = Logger("ui") 