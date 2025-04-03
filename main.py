import sys
import os
import traceback
import logging
import atexit
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow
from ui.theme_manager import ThemeManager
from ui.splash_screen import SplashScreen
from ui.app_icon import generate_app_icon

from models import app_logger
from models.lock_manager import lock_manager

# 全局变量
app = None
main_window = None
splash = None

def initialize_app_resources():
    """初始化应用资源"""
    # 创建应用图标
    try:
        # 检查图标目录中是否有自定义图标
        icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'icons')
        main_icon_path = os.path.join(icons_dir, 'app_icon.png')
        
        if os.path.exists(main_icon_path):
            app_logger.info(f"使用自定义图标: {main_icon_path}")
        else:
            app_logger.info("未找到自定义图标，将生成默认图标")
            
        app_icon = generate_app_icon()
        if app_icon:
            app.setWindowIcon(app_icon)
    except Exception as e:
        app_logger.error(f"创建应用图标失败: {str(e)}")

def cleanup_resources():
    """清理应用资源"""
    # 清理锁资源
    app_logger.info("应用程序退出，清理锁资源...")
    try:
        lock_manager.clear_expired_locks()
    except Exception as e:
        app_logger.error(f"清理锁资源时出错: {str(e)}")

def launch_main_window():
    """启动主窗口"""
    global main_window
    
    # 创建主窗口
    if main_window is None:
        main_window = MainWindow()
        main_window.show()
        
    # 关闭启动画面
    if splash:
        splash.finish(main_window)

def handle_exception(exc_type, exc_value, exc_traceback):
    """处理未捕获的异常"""
    # 忽略KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    # 记录异常
    app_logger.critical("未捕获的异常", 
                     exc_info=(exc_type, exc_value, exc_traceback))
    
    # 显示异常对话框
    error_msg = f"发生未处理的异常:\n{exc_value}\n\n详细信息已记录到日志中。"
    if app:
        QMessageBox.critical(None, "错误", error_msg)
    else:
        print(error_msg)

def main():
    """程序入口"""
    global app, splash
    
    # 注册异常处理器
    sys.excepthook = handle_exception
    
    # 注册退出清理函数
    atexit.register(cleanup_resources)
    
    # 创建应用实例
    app = QApplication(sys.argv)
    app.setApplicationName("AI Chat Model")
    
    # 创建启动画面
    splash = SplashScreen()
    splash.show()
    
    # 应用主题
    theme_manager = ThemeManager()
    theme_manager.apply_theme(app)
    
    # 初始化资源
    initialize_app_resources()
    
    # 启动主窗口
    splash.set_progress(100)
    launch_main_window()
    
    # 进入事件循环
    return app.exec_()

if __name__ == "__main__":
    # 尝试导入演示脚本
    demo_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "run_demo.py")
    
    try:
        # 检查是否存在演示脚本
        if os.path.exists(demo_script_path):
            from scripts.run_demo import main as run_demo
            # 运行演示
            sys.exit(run_demo())
        else:
            # 运行正常应用
            sys.exit(main())
    except ImportError:
        # 运行正常应用
        sys.exit(main()) 