"""
主题管理器模块
负责处理应用程序的样式和主题
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt
import os
import json
import sys

# 确保models目录在sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.resource_manager import ResourceManager
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.resource_manager import ResourceManager

class ThemeManager:
    """主题管理器，负责处理应用程序的样式和主题"""
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化主题管理器"""
        if self._initialized:
            return
            
        # 获取资源管理器
        self.resource_manager = ResourceManager()
        
        # 基本路径设置
        self.ui_dir = os.path.dirname(os.path.abspath(__file__))
        self.styles_path = os.path.join(self.ui_dir, 'styles.qss')
        
        # 主题定义
        self.themes = {
            "light": {
                "name": "默认浅色",
                "primary_color": "#0078D7",
                "secondary_color": "#E6F7FF",
                "background_color": "#FFFFFF",
                "text_color": "#333333",
                "success_color": "#28A745",
                "warning_color": "#FFC107",
                "error_color": "#DC3545"
            },
            "dark": {
                "name": "深色模式",
                "primary_color": "#0078D7",
                "secondary_color": "#2C2C2C",
                "background_color": "#1E1E1E",
                "text_color": "#FFFFFF",
                "success_color": "#28A745",
                "warning_color": "#FFC107",
                "error_color": "#DC3545"
            },
            "blue": {
                "name": "蓝色主题",
                "primary_color": "#2196F3",
                "secondary_color": "#E3F2FD",
                "background_color": "#F5F9FC",
                "text_color": "#263238",
                "success_color": "#4CAF50",
                "warning_color": "#FF9800",
                "error_color": "#F44336"
            },
            "green": {
                "name": "绿色主题",
                "primary_color": "#4CAF50",
                "secondary_color": "#E8F5E9",
                "background_color": "#F5F9F6",
                "text_color": "#1B5E20",
                "success_color": "#2E7D32",
                "warning_color": "#FF9800",
                "error_color": "#F44336"
            }
        }
        
        # 从配置中加载当前主题
        ui_config = self.resource_manager.get_ui_config()
        self.current_theme = ui_config.get("theme", "light")
        if self.current_theme not in self.themes:
            self.current_theme = "light"
        
        # 字体设置
        self.font_family = ui_config.get("font_family", "Microsoft YaHei")
        self.font_size = ui_config.get("font_size", 10)
        
        self._initialized = True
    
    def get_available_themes(self):
        """获取可用的主题列表"""
        return [(theme_id, theme_data["name"]) for theme_id, theme_data in self.themes.items()]
    
    def get_current_theme(self):
        """获取当前主题ID"""
        return self.current_theme
    
    def set_theme(self, theme_id):
        """设置当前主题
        
        Args:
            theme_id: 主题ID
            
        Returns:
            bool: 设置是否成功
        """
        if theme_id not in self.themes:
            return False
        
        self.current_theme = theme_id
        
        # 更新配置
        ui_config = self.resource_manager.get_ui_config()
        ui_config["theme"] = theme_id
        self.resource_manager.config["ui"] = ui_config
        self.resource_manager.save_config()
        
        return True
    
    def apply_theme(self, app):
        """应用当前主题到应用程序
        
        Args:
            app: QApplication实例
        """
        theme_data = self.themes[self.current_theme]
        
        # 应用QSS样式
        qss = self._generate_stylesheet(theme_data)
        app.setStyleSheet(qss)
        
        # 设置应用程序字体
        font = QFont(self.font_family, self.font_size)
        app.setFont(font)
        
        # 应用调色板
        self._apply_palette(app, theme_data)
    
    def _generate_stylesheet(self, theme_data):
        """生成QSS样式表
        
        Args:
            theme_data: 主题数据
            
        Returns:
            str: QSS样式表
        """
        # 读取基础样式文件
        base_style = ""
        if os.path.exists(self.styles_path):
            with open(self.styles_path, 'r', encoding='utf-8') as f:
                base_style = f.read()
        
        # 替换颜色变量
        style = base_style.replace("$PRIMARY_COLOR", theme_data["primary_color"])
        style = style.replace("$SECONDARY_COLOR", theme_data["secondary_color"])
        style = style.replace("$BACKGROUND_COLOR", theme_data["background_color"])
        style = style.replace("$TEXT_COLOR", theme_data["text_color"])
        style = style.replace("$SUCCESS_COLOR", theme_data["success_color"])
        style = style.replace("$WARNING_COLOR", theme_data["warning_color"])
        style = style.replace("$ERROR_COLOR", theme_data["error_color"])
        
        # 添加全局样式
        additional_style = f"""
        QWidget {{
            background-color: {theme_data["background_color"]};
            color: {theme_data["text_color"]};
            font-family: {self.font_family};
            font-size: {self.font_size}pt;
        }}
        
        QPushButton {{
            background-color: {theme_data["primary_color"]};
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
        }}
        
        QPushButton:hover {{
            background-color: {self._lighten_color(theme_data["primary_color"], 20)};
        }}
        
        QPushButton:pressed {{
            background-color: {self._darken_color(theme_data["primary_color"], 20)};
        }}
        
        QPushButton:disabled {{
            background-color: #B0B0B0;
            color: #686868;
        }}
        
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {self._adjust_background(theme_data["background_color"])};
            color: {theme_data["text_color"]};
            border: 1px solid #B0B0B0;
            border-radius: 4px;
            padding: 2px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid #B0B0B0;
            border-radius: 4px;
        }}
        
        QTabBar::tab {{
            background-color: {self._lighten_color(theme_data["secondary_color"], 10)};
            color: {theme_data["text_color"]};
            padding: 6px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {theme_data["primary_color"]};
            color: white;
        }}
        
        QGroupBox {{
            border: 1px solid #B0B0B0;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 15px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: {theme_data["background_color"]};
        }}
        
        QProgressBar {{
            border: 1px solid #B0B0B0;
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background-color: {theme_data["primary_color"]};
            width: 5px;
        }}
        
        QMenu {{
            background-color: {theme_data["background_color"]};
            border: 1px solid #B0B0B0;
        }}
        
        QMenu::item {{
            padding: 5px 25px 5px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {theme_data["secondary_color"]};
        }}
        
        QHeaderView::section {{
            background-color: {theme_data["secondary_color"]};
            color: {theme_data["text_color"]};
            padding: 5px;
            border: 1px solid #B0B0B0;
        }}
        
        QScrollBar:vertical {{
            border: none;
            background-color: {self._adjust_background(theme_data["background_color"])};
            width: 12px;
            margin: 12px 0px 12px 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self._lighten_color(theme_data["primary_color"], 50)};
            min-height: 25px;
            border-radius: 4px;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 12px;
        }}
        """
        
        return style + additional_style
    
    def _apply_palette(self, app, theme_data):
        """应用调色板
        
        Args:
            app: QApplication实例
            theme_data: 主题数据
        """
        palette = QPalette()
        
        # 窗口和控件背景色
        palette.setColor(QPalette.Window, QColor(theme_data["background_color"]))
        palette.setColor(QPalette.WindowText, QColor(theme_data["text_color"]))
        palette.setColor(QPalette.Base, QColor(self._adjust_background(theme_data["background_color"])))
        palette.setColor(QPalette.AlternateBase, QColor(self._adjust_background(theme_data["background_color"], 10)))
        
        # 文本颜色
        palette.setColor(QPalette.Text, QColor(theme_data["text_color"]))
        palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
        
        # 按钮颜色
        palette.setColor(QPalette.Button, QColor(theme_data["primary_color"]))
        palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
        
        # 高亮和链接颜色
        palette.setColor(QPalette.Highlight, QColor(theme_data["primary_color"]))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        palette.setColor(QPalette.Link, QColor(theme_data["primary_color"]))
        palette.setColor(QPalette.LinkVisited, QColor(self._darken_color(theme_data["primary_color"], 20)))
        
        # 禁用状态颜色
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#787878"))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#787878"))
        
        app.setPalette(palette)
    
    def _adjust_background(self, color, adjust=0):
        """调整背景颜色
        
        Args:
            color: 原始颜色
            adjust: 调整量
            
        Returns:
            str: 调整后的颜色
        """
        if color.startswith("#"):
            color = color[1:]
        
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        
        # 判断是浅色还是深色主题
        is_dark = (r * 0.299 + g * 0.587 + b * 0.114) < 128
        
        if is_dark:
            # 深色主题，略微提亮背景
            r = min(255, r + 10 + adjust)
            g = min(255, g + 10 + adjust)
            b = min(255, b + 10 + adjust)
        else:
            # 浅色主题，略微降低背景亮度
            r = max(0, r - 5 - adjust)
            g = max(0, g - 5 - adjust)
            b = max(0, b - 5 - adjust)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def _lighten_color(self, color, percent):
        """提亮颜色
        
        Args:
            color: 原始颜色
            percent: 提亮百分比
            
        Returns:
            str: 提亮后的颜色
        """
        if color.startswith("#"):
            color = color[1:]
        
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        
        r = min(255, r + (255 - r) * percent // 100)
        g = min(255, g + (255 - g) * percent // 100)
        b = min(255, b + (255 - b) * percent // 100)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def _darken_color(self, color, percent):
        """加深颜色
        
        Args:
            color: 原始颜色
            percent: 加深百分比
            
        Returns:
            str: 加深后的颜色
        """
        if color.startswith("#"):
            color = color[1:]
        
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        
        r = max(0, r - r * percent // 100)
        g = max(0, g - g * percent // 100)
        b = max(0, b - b * percent // 100)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def apply_current_theme(self):
        """应用当前主题到应用程序"""
        app = QApplication.instance()
        if app:
            self.apply_theme(app)
    
    def toggle_theme(self):
        """切换主题
        
        在所有可用主题间轮流切换
        """
        # 获取所有主题的ID列表
        theme_ids = list(self.themes.keys())
        
        # 找到当前主题的索引
        current_index = theme_ids.index(self.current_theme)
        
        # 计算下一个主题的索引（循环）
        next_index = (current_index + 1) % len(theme_ids)
        
        # 设置为下一个主题
        next_theme = theme_ids[next_index]
        self.set_theme(next_theme)
        
        # 应用新主题
        self.apply_current_theme()
        
        # 返回新主题的名称，用于显示
        return self.themes[next_theme]["name"] 