import sys
import os
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    # 导入运行demo的脚本
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    sys.path.insert(0, scripts_dir)
    
    try:
        # 尝试导入并运行demo脚本的main函数
        from scripts.run_demo import main as run_demo_main
        run_demo_main()
    except ImportError:
        # 如果脚本导入失败，则直接启动应用
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_()) 