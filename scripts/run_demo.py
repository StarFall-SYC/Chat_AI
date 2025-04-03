#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示运行脚本
可以从命令行直接运行此脚本以启动应用程序
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 添加父目录到模块搜索路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """
    主函数 - 运行演示应用
    """
    from ui.main_window import MainWindow
    from ui.theme_manager import ThemeManager
    
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 设置应用元数据
    app.setApplicationName("AI聊天大模型")
    app.setApplicationVersion("1.2.0")
    app.setOrganizationName("AI技术团队")
    
    # 应用主题
    theme_manager = ThemeManager()
    theme_manager.apply_theme(app)
    
    # 创建并显示主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 