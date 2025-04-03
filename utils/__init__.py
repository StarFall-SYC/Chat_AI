"""
工具函数模块
包含各类辅助函数
"""

import os
import sys

# 确保models目录在sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import app_logger
except ImportError:
    # 创建占位符
    class DummyLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    app_logger = DummyLogger()


def get_app_icon_path():
    """获取应用程序图标路径

    返回应用程序图标的完整路径，优先使用高分辨率图标

    Returns:
        str: 图标文件路径
    """
    # 获取项目根目录
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 定义可能的图标路径，按优先级排序
    icon_paths = [
        os.path.join(base_path, 'ui', 'icons', 'app_icon.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_256.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_128.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_64.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_48.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_32.png'),
        os.path.join(base_path, 'ui', 'icons', 'app_icon_16.png'),
    ]
    
    # 检查图标是否存在
    for path in icon_paths:
        if os.path.exists(path):
            app_logger.info(f"找到应用图标: {path}")
            return path
    
    # 如果找不到图标，返回默认图标路径
    app_logger.warning("找不到应用图标，将使用默认路径。请检查ui/icons目录中是否有图标文件。")
    app_logger.info("您可以在ui/icons目录中放置自定义图标。详见ui/icons/CUSTOM_ICONS.md")
    return icon_paths[0] 