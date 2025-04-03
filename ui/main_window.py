"""
主窗口模块定义
创建和管理整个应用程序的主窗口
"""

import os
import sys
import time
import atexit

from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QTextEdit, QAction, 
                           QTabWidget, QToolBar, QToolButton, QMenu, 
                           QWidget, QMessageBox, QSplitter, QStatusBar, 
                           QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, QTimer, QTime
from PyQt5.QtGui import QFont, QIcon, QPixmap

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import app_logger
from models.lock_manager import LockManager
from utils import get_app_icon_path

# 导入基本选项卡
from .tabs import ChatTab

# 导入我们新创建的模块
from .model_training import ModelTrainingTab
from .data_analysis import DataAnalysisTab
from .theme_manager import ThemeManager

# 检查是否可使用高级模型
try:
    from .advanced_tabs import ImageGenerationTab, VideoGenerationTab
    from models import advanced_models
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    app_logger.warning("高级模型功能不可用，将使用基本功能")
    ADVANCED_MODELS_AVAILABLE = False

from .custom_widgets import (
    ChatMessageList, AnimatedButton, FloatingActionButton, 
    ModernButton, ModernToolButton, StyledCard, StatusInfoWidget
)


class AppState:
    """应用程序状态类，用于管理全局变量"""
    
    def __init__(self):
        """初始化应用程序状态"""
        self.model = None
        self.transformer_model = None
        self.lock_manager = LockManager()


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 应用状态
        self.app_state = AppState()
        
        # 防止重复加载
        self.models_loaded = False
        
        # 设置窗口基本属性
        self.setWindowTitle("AI聊天模型")
        self.setMinimumSize(800, 600)
        self.setWindowIcon(QIcon(get_app_icon_path()))
        
        # 上次激活的高级功能标签索引
        self.last_active_advanced_tab = 0
        
        # 初始化UI
        self.setup_ui()
        
        # 设置工具栏和状态栏
        self.setup_toolbar()
        self.setup_statusbar()
        
        # 加载模型（延迟加载）
        QTimer.singleShot(500, self.setup_models)
        
        # 设置主题
        self.theme_manager = ThemeManager()
        self.theme_manager.apply_current_theme()
        
        # 清理锁管理器资源
        atexit.register(self.cleanup_resources)
    
    def setup_ui(self):
        """设置UI"""
        # 设置标签页
        self.setup_tabs()
    
    def setup_tabs(self):
        """设置选项卡"""
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(True)
        
        # 隐藏标签页标题栏，因为我们使用工具栏按钮导航
        self.tab_widget.tabBar().setVisible(False)
        
        # 创建聊天选项卡
        self.chat_tab = ChatTab(self)
        self.tab_widget.addTab(self.chat_tab, "AI聊天")
        
        # 检查是否应该加载高级功能
        # 如果系统支持高级功能，加载高级选项卡
        if ADVANCED_MODELS_AVAILABLE:
            # 创建图像生成选项卡
            self.image_tab = ImageGenerationTab(self)
            self.tab_widget.addTab(self.image_tab, "图像生成")
            
            # 创建视频生成选项卡
            self.video_tab = VideoGenerationTab(self)
            self.tab_widget.addTab(self.video_tab, "视频生成")
            
            # 添加模型训练选项卡
            self.training_tab = ModelTrainingTab(self)
            self.tab_widget.addTab(self.training_tab, "模型训练")
            
            # 添加数据分析选项卡
            self.analysis_tab = DataAnalysisTab(self)
            self.tab_widget.addTab(self.analysis_tab, "数据分析")
        else:
            # 添加仅包含基本模型的模型训练选项卡
            self.training_tab = ModelTrainingTab(self)
            self.tab_widget.addTab(self.training_tab, "模型训练")
            
            # 添加数据分析选项卡
            self.analysis_tab = DataAnalysisTab(self)
            self.tab_widget.addTab(self.analysis_tab, "数据分析")
            
        # 连接标签切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 设置为中心部件
        self.setCentralWidget(self.tab_widget)
    
    def setup_toolbar(self):
        """设置工具栏"""
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setIconSize(QSize(28, 28))
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(self.toolbar)
        
        # 增加背景色标识当前标签页的功能
        self.toolButtons = []
        
        # 添加聊天按钮
        chat_button = ModernToolButton()
        chat_button.setText("AI聊天")
        chat_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        chat_button.setCheckable(True)
        chat_button.setChecked(True)  # 默认选中
        chat_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        self.toolbar.addWidget(chat_button)
        self.toolButtons.append(chat_button)
        
        # 如果有高级功能
        if ADVANCED_MODELS_AVAILABLE:
            # 添加图像生成按钮
            image_button = ModernToolButton()
            image_button.setText("图像生成")
            image_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            image_button.setCheckable(True)
            image_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
            self.toolbar.addWidget(image_button)
            self.toolButtons.append(image_button)
            
            # 添加视频生成按钮
            video_button = ModernToolButton()
            video_button.setText("视频生成")
            video_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            video_button.setCheckable(True)
            video_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
            self.toolbar.addWidget(video_button)
            self.toolButtons.append(video_button)
            
            # 添加训练按钮
            training_button = ModernToolButton()
            training_button.setText("模型训练")
            training_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            training_button.setCheckable(True)
            training_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
            self.toolbar.addWidget(training_button)
            self.toolButtons.append(training_button)
            
            # 添加数据分析按钮
            analysis_button = ModernToolButton()
            analysis_button.setText("数据分析")
            analysis_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            analysis_button.setCheckable(True)
            analysis_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(4))
            self.toolbar.addWidget(analysis_button)
            self.toolButtons.append(analysis_button)
        else:
            # 添加训练按钮
            training_button = ModernToolButton()
            training_button.setText("模型训练")
            training_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            training_button.setCheckable(True)
            training_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
            self.toolbar.addWidget(training_button)
            self.toolButtons.append(training_button)
            
            # 添加数据分析按钮
            analysis_button = ModernToolButton()
            analysis_button.setText("数据分析")
            analysis_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            analysis_button.setCheckable(True)
            analysis_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
            self.toolbar.addWidget(analysis_button)
            self.toolButtons.append(analysis_button)
        
        # 添加弹簧(占位符)，使尾部按钮靠右显示
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # 添加清除聊天按钮
        clear_chat_button = ModernToolButton()
        clear_chat_button.setText("清除聊天")
        clear_chat_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        clear_chat_button.clicked.connect(lambda: self.tab_widget.setCurrentWidget(self.chat_tab) or self.chat_tab.clear_chat())
        self.toolbar.addWidget(clear_chat_button)
        
        # 添加帮助按钮
        help_button = ModernToolButton()
        help_button.setText("帮助")
        help_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        help_button.clicked.connect(self.show_about)
        self.toolbar.addWidget(help_button)
        
        # 添加主题切换按钮
        theme_button = ModernToolButton()
        theme_button.setText("切换主题")
        theme_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        theme_button.clicked.connect(self.toggle_theme)
        self.toolbar.addWidget(theme_button)
    
    def setup_statusbar(self):
        """设置状态栏"""
        self.statusBar().showMessage("就绪")
        
        # 添加状态信息标签
        self.status_info = QLabel()
        self.statusBar().addPermanentWidget(self.status_info)
        
        # 添加状态图标
        self.status_icon = QLabel()
        status_pixmap = QPixmap(16, 16)
        status_pixmap.fill(Qt.green)
        self.status_icon.setPixmap(status_pixmap)
        self.statusBar().addPermanentWidget(self.status_icon)
        
        # 添加时间标签
        self.time_label = QLabel()
        self.statusBar().addPermanentWidget(self.time_label)
        
        # 启动定时器，每秒更新一次时间
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_time)
        self.status_timer.start(1000)  # 每秒更新一次
    
    def update_status_time(self):
        """更新状态栏时间"""
        current_time = QTime.currentTime()
        time_text = current_time.toString("hh:mm:ss")
        self.time_label.setText(time_text)
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 悬浮按钮已移除，这里保留空实现以便后续扩展
    
    def setup_models(self):
        """设置模型"""
        # 加载模型
        self.load_model()
        
        # 如果高级模型可用，加载高级功能
        if ADVANCED_MODELS_AVAILABLE:
            self.load_transformer_model()
    
    def load_model(self):
        """加载基础模型"""
        if self.models_loaded:
            return
            
        try:
            self.statusBar().showMessage("正在加载模型...")
            
            # 这里实现模型加载逻辑
            # 实际应用中，应该导入并初始化相应的模型
            
            self.models_loaded = True
            self.statusBar().showMessage("模型加载完成", 3000)
        except Exception as e:
            app_logger.error(f"加载模型失败: {str(e)}")
            self.statusBar().showMessage("模型加载失败", 3000)
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
    
    def load_transformer_model(self):
        """加载Transformer模型"""
        try:
            self.statusBar().showMessage("正在加载高级模型...")
            
            # 这里实现高级模型加载逻辑
            # 实际应用中，应该导入并初始化相应的模型
            
            self.statusBar().showMessage("高级模型加载完成", 3000)
        except Exception as e:
            app_logger.error(f"加载高级模型失败: {str(e)}")
            self.statusBar().showMessage("高级模型加载失败", 3000)
            QMessageBox.critical(self, "错误", f"加载高级模型失败: {str(e)}")
    
    def cleanup_resources(self):
        """清理资源"""
        # 清理锁资源
        if hasattr(self, 'app_state') and hasattr(self.app_state, 'lock_manager'):
            # 调用正确的方法名
            if hasattr(self.app_state.lock_manager, 'cleanup_all_locks'):
                self.app_state.lock_manager.cleanup_all_locks()
            elif hasattr(self.app_state.lock_manager, 'clear_expired_locks'):
                self.app_state.lock_manager.clear_expired_locks()
            app_logger.info("已清理所有锁资源")
    
    def toggle_theme(self):
        """切换主题"""
        self.theme_manager.toggle_theme()
    
    def show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "设置", "设置功能即将上线")
    
    def on_tab_changed(self, index):
        """标签页切换事件"""
        # 更新工具栏按钮状态
        for i, button in enumerate(self.toolButtons):
            # 确保按钮状态与当前标签页索引匹配
            button.setChecked(i == index)
                
        # 添加其他标签页切换逻辑
        if ADVANCED_MODELS_AVAILABLE:
            if index >= 1 and index <= 2:  # 图像或视频生成页
                self.last_active_advanced_tab = index
    
    def show_about(self):
        """显示关于对话框"""
        about_dialog = AboutDialog(self)
        about_dialog.exec_()


class AboutDialog(QMessageBox):
    """关于对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setWindowIcon(QIcon(get_app_icon_path()))
        self.setIconPixmap(QPixmap(get_app_icon_path()).scaled(64, 64))
        
        # 设置文本
        self.setText("<h3>AI聊天模型</h3>")
        self.setInformativeText(
            "一个强大的AI聊天应用程序，提供了丰富的功能，包括聊天、训练、图像生成等。\n\n"
            "版本: 1.0.0\n"
            "构建时间: 2023.4.15\n\n"
            "© 2023 AI开发团队. 保留所有权利。"
        )
        
        # 设置详细文本
        self.setDetailedText(
            "本应用程序使用了以下技术:\n"
            "- PyQt5\n"
            "- Transformer架构\n"
            "- 自然语言处理技术\n"
            "- 机器学习算法\n\n"
            "感谢所有为本项目做出贡献的开发者。"
        )
        
        # 设置标准按钮
        self.setStandardButtons(QMessageBox.Ok) 