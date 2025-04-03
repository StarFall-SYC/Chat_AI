"""
UI模块包，包含所有与用户界面相关的组件
"""

from .main_window import MainWindow
from .splash_screen import SplashScreen
from .theme_manager import ThemeManager
from .model_training import ModelTrainingTab
from .data_analysis import DataAnalysisTab

__all__ = ['MainWindow', 'SplashScreen', 'ThemeManager', 'ModelTrainingTab', 'DataAnalysisTab']

# 导出主要组件
from .main_window import MainWindow
from .tabs import ChatTab, TrainingTab
# 导出高级标签页组件
try:
    from .advanced_tabs import MediaGenerationTab, ImageGenerationTab, VideoGenerationTab
except ImportError:
    pass 