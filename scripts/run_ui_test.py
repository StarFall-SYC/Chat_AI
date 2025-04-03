#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI测试脚本
用于测试我们重写的UI组件
"""

import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget
from PyQt5.QtCore import Qt, QTimer

# 添加父目录到模块搜索路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 从UI组件和主题管理器导入
from ui.custom_widgets import ChatBubble, ChatMessageList, AnimatedButton
from ui.tabs import ChatTab
from ui.splash_screen import SplashScreen
from ui.theme_manager import ThemeManager


class UITestWindow(QMainWindow):
    """UI测试窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("AI聊天大模型 - UI测试")
        self.resize(960, 720)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 添加聊天标签页
        self.chat_tab = ChatTab()
        tab_widget.addTab(self.chat_tab, "聊天")
        
        # 添加组件测试标签页
        self.components_tab = self.create_components_test_tab()
        tab_widget.addTab(self.components_tab, "组件测试")
        
        layout.addWidget(tab_widget)
        
        # 测试消息定时器
        self.test_message_timer = QTimer(self)
        self.test_message_timer.timeout.connect(self.add_test_message)
        self.test_message_timer.start(5000)  # 每5秒添加一条测试消息
        
        # 测试计数器
        self.test_count = 0
    
    def create_components_test_tab(self):
        """创建组件测试标签页"""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # 添加测试消息列表
        self.test_message_list = ChatMessageList()
        layout.addWidget(self.test_message_list)
        
        # 添加测试按钮
        test_button = AnimatedButton("添加测试消息")
        test_button.clicked.connect(self.add_test_message)
        layout.addWidget(test_button)
        
        return container
    
    def add_test_message(self):
        """添加测试消息"""
        self.test_count += 1
        
        if self.test_count % 2 == 0:
            # 用户消息
            self.test_message_list.add_message("user", f"这是测试用户消息 #{self.test_count}，测试UI组件的渲染效果。")
        else:
            # 助手消息
            self.test_message_list.add_message("assistant", f"这是测试助手消息 #{self.test_count}，用于测试改进后的UI组件。包含一些较长的文本来测试换行和自动调整高度的功能。")
            
        # 每5条消息添加一条带图片的消息
        if self.test_count % 5 == 0:
            # 尝试查找示例图片
            image_paths = [
                os.path.join(parent_dir, "data", "generated", "images", "test_image.jpg"),
                os.path.join(parent_dir, "data", "generated", "images", "sample.png"),
                os.path.join(parent_dir, "resources", "images", "sample.jpg")
            ]
            
            image_path = None
            for path in image_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path:
                self.test_message_list.add_message("assistant", "这是一条带图片的测试消息", media_path=image_path, media_type="image")
            else:
                # 找不到图片时，创建测试图片目录并留言
                test_dir = os.path.join(parent_dir, "data", "generated", "images")
                os.makedirs(test_dir, exist_ok=True)
                self.test_message_list.add_message("assistant", f"找不到测试图片，请在 {test_dir} 目录下添加名为test_image.jpg的图片进行测试")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用元数据
    app.setApplicationName("AI聊天UI测试")
    app.setApplicationVersion("1.0.0")
    
    # 应用主题
    theme_manager = ThemeManager()
    theme_manager.apply_theme(app)
    
    # 创建并显示启动画面
    splash = SplashScreen()
    
    # 主窗口
    main_window = UITestWindow()
    
    # 设置启动画面关闭后显示主窗口
    def show_main():
        main_window.show()
        splash.hide()  # 确保隐藏启动屏幕
    
    # 启动加载动画，设置完成后的回调函数
    splash.start_animation(on_finished=show_main)
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 