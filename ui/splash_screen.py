"""
启动屏幕模块
显示应用程序启动时的闪屏
"""

from PyQt5.QtWidgets import (QSplashScreen, QProgressBar, QVBoxLayout, 
                           QLabel, QWidget, QApplication)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QLinearGradient, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, QSize, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty

import time
import os


class SplashScreen(QSplashScreen):
    """自定义启动屏幕，带进度条和动画效果"""
    
    def __init__(self, parent=None):
        """初始化启动屏幕"""
        # 设置尺寸
        self.width = 600
        self.height = 400
        
        # 确保使用高DPI显示
        self.pixmap_ratio = QApplication.instance().devicePixelRatio()
        pixmap = QPixmap(int(self.width * self.pixmap_ratio), int(self.height * self.pixmap_ratio))
        pixmap.setDevicePixelRatio(self.pixmap_ratio)
        pixmap.fill(Qt.transparent)  # 填充透明背景
        
        # 在pixmap上预先绘制背景
        self._draw_background(pixmap)
        
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # 确保始终在顶部
        self.setWindowFlag(Qt.FramelessWindowHint)   # 无边框窗口
        
        # 创建容器窗口小部件
        self.content_widget = QWidget(self)
        self.content_widget.setGeometry(0, 0, self.width, self.height)
        self.content_widget.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        
        # 注册字体（如果有自定义字体）
        # self._register_fonts()
        
        # 创建垂直布局
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(10)
        
        # 添加顶部空间
        self.layout.addSpacing(20)
        
        # 创建标题标签
        self.title_label = QLabel("AI聊天大模型")
        self.title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #1565C0; background: transparent;")
        self.layout.addWidget(self.title_label)
        
        # 创建副标题
        self.subtitle_label = QLabel("智能对话  专业解答  多模态交互")
        self.subtitle_label.setFont(QFont("Microsoft YaHei", 12))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #424242; background: transparent;")
        self.layout.addWidget(self.subtitle_label)
        
        # 添加垂直间距
        self.layout.addSpacing(50)
        
        # 添加加载消息标签
        self.message_label = QLabel("正在初始化应用程序...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #616161; background: transparent; font-size: 11pt;")
        self.layout.addWidget(self.message_label)
        
        # 添加一个小间距
        self.layout.addSpacing(6)
        
        # 创建进度条容器（为了添加左右边距）
        progress_container = QWidget()
        progress_container.setStyleSheet("background: transparent;")
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(60, 0, 60, 0)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: rgba(224, 224, 224, 180);
                text-align: center;
                color: transparent;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                stop:0 #1976D2, stop:1 #42A5F5);
                border-radius: 6px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        self.layout.addWidget(progress_container)
        
        # 添加弹性空间
        self.layout.addStretch(1)
        
        # 添加版本信息
        self.version_label = QLabel("版本 1.2.0")
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet("color: #9E9E9E; font-size: 9pt; background: transparent;")
        self.layout.addWidget(self.version_label)
        
        # 设置进度和消息列表
        self.progress_steps = [
            ("正在初始化应用程序...", 10),
            ("加载UI组件...", 30),
            ("准备聊天模型...", 50),
            ("检查系统资源...", 70),
            ("连接到服务...", 90),
            ("准备就绪！", 100)
        ]
        
        # 初始化进度计数器
        self.progress_index = 0
        self.current_progress = 0
        self.target_progress = 0
        
        # 设置动画
        self._progress_value = 0
        self.progress_animation = QPropertyAnimation(self, b"progress_value")
        self.progress_animation.setDuration(600)  # 600毫秒动画
        self.progress_animation.setEasingCurve(QEasingCurve.OutQuad)
        # 连接动画值变化信号
        self.progress_animation.valueChanged.connect(self._update_progress_display)
        
        # 设置步骤计时器
        self.step_timer = QTimer(self)
        self.step_timer.timeout.connect(self._next_progress_step)
        
        # 初始化颜色动画
        self.color_timer = QTimer(self)
        self.color_timer.timeout.connect(self._update_colors)
        self.color_timer.start(3000)  # 每3秒变换一次颜色
        self.color_shift = 0
        
        # 显示在屏幕中央
        self.center_on_screen()
    
    def _register_fonts(self):
        """注册自定义字体"""
        # 例如添加一个漂亮的字体作为标题
        font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'fonts')
        os.makedirs(font_dir, exist_ok=True)
        
        # 检查几种可能的字体文件
        font_paths = [
            os.path.join(font_dir, 'OpenSans-Bold.ttf'),
            os.path.join(font_dir, 'SourceHanSansCN-Bold.otf')
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                QFontDatabase.addApplicationFont(font_path)
    
    def _update_colors(self):
        """更新渐变颜色动画"""
        self.color_shift = (self.color_shift + 1) % 6
        
        # 颜色组合列表
        color_sets = [
            ("#1976D2", "#42A5F5"),  # 蓝色
            ("#0097A7", "#26C6DA"),  # 青色
            ("#00796B", "#26A69A"),  # 绿松石
            ("#00ACC1", "#4DD0E1"),  # 浅青色
            ("#3949AB", "#5C6BC0"),  # 靛蓝色
            ("#5E35B1", "#7E57C2")   # 深紫色
        ]
        
        color1, color2 = color_sets[self.color_shift]
        
        # 更新进度条颜色
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 6px;
                background-color: rgba(224, 224, 224, 180);
                text-align: center;
                color: transparent;
            }}
            
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                stop:0 {color1}, stop:1 {color2});
                border-radius: 6px;
            }}
        """)
        
        # 同时更新标题颜色
        self.title_label.setStyleSheet(f"color: {color1}; background: transparent;")
    
    def center_on_screen(self):
        """将启动屏幕居中显示"""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width) // 2
        y = (screen.height() - self.height) // 2
        self.move(x, y)
    
    def _draw_background(self, pixmap):
        """预先在pixmap上绘制背景
        
        Args:
            pixmap: 要绘制的pixmap
        """
        painter = QPainter()
        painter.begin(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制圆角矩形背景
        rect = pixmap.rect()
        painter.setPen(Qt.NoPen)
        
        # 创建渐变背景 - 使用更微妙的渐变
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 250))
        gradient.setColorAt(1, QColor(245, 245, 245, 250))
        
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 15 * self.pixmap_ratio, 15 * self.pixmap_ratio)
        
        # 绘制精致边框
        painter.setPen(QColor(200, 200, 200, 150))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 
                             15 * self.pixmap_ratio, 
                             15 * self.pixmap_ratio)
        
        # 添加底部装饰性条纹
        accent_gradient = QLinearGradient(0, rect.height() - 40 * self.pixmap_ratio, 
                                       rect.width(), 
                                       rect.height() - 40 * self.pixmap_ratio)
        accent_gradient.setColorAt(0, QColor(25, 118, 210, 40))   # 蓝色透明
        accent_gradient.setColorAt(0.5, QColor(66, 165, 245, 60)) # 浅蓝透明
        accent_gradient.setColorAt(1, QColor(25, 118, 210, 40))   # 蓝色透明
        
        painter.setBrush(accent_gradient)
        painter.setPen(Qt.NoPen)
        bottom_rect = QRect(0, 
                         int(rect.height() - 40 * self.pixmap_ratio), 
                         rect.width(), 
                         int(40 * self.pixmap_ratio))
        painter.drawRect(bottom_rect)
        
        painter.end()
    
    @pyqtProperty(float)
    def progress_value(self):
        """获取当前进度值"""
        return self._progress_value
    
    @progress_value.setter
    def progress_value(self, value):
        """设置当前进度值"""
        self._progress_value = value
        self._update_progress_display(value)
    
    def _update_progress_display(self, value):
        """更新进度条显示"""
        if value is not None:
            self.progress_bar.setValue(int(value))
            QApplication.processEvents()
    
    def start_animation(self, on_finished=None):
        """开始加载动画
        
        Args:
            on_finished: 动画完成时的回调函数
        """
        self.on_finished_callback = on_finished
        
        # 重置状态
        self.progress_index = 0
        self._progress_value = 0
        self.target_progress = 0
        
        # 获取第一步的消息和进度值
        if self.progress_index < len(self.progress_steps):
            message, progress = self.progress_steps[self.progress_index]
            self.message_label.setText(message)
            self.target_progress = progress
        
        # 设置初始进度动画
        self.progress_animation.setStartValue(self._progress_value)
        self.progress_animation.setEndValue(self.target_progress)
        self.progress_animation.start()
        
        # 启动步骤计时器
        self.step_timer.start(1500)  # 每1.5秒更新一次步骤
        
        # 显示启动屏幕
        self.show()
        
        # 确保立即应用UI更新
        QApplication.processEvents()
    
    def _next_progress_step(self):
        """推进到下一个步骤"""
        self.progress_index += 1
        
        # 检查是否完成所有步骤
        if self.progress_index >= len(self.progress_steps):
            # 停止定时器
            self.step_timer.stop()
            self.color_timer.stop()
            
            # 如果有回调，执行回调
            if hasattr(self, 'on_finished_callback') and self.on_finished_callback:
                # 更新消息和进度条
                self.message_label.setText("加载完成!")
                self.progress_bar.setValue(100)
                QApplication.processEvents()
                
                # 短暂延迟后执行回调
                QTimer.singleShot(800, self._call_finished_callback)
            return
        
        # 获取下一步的消息和进度
        message, progress = self.progress_steps[self.progress_index]
        self.message_label.setText(message)
        
        # 设置目标进度
        self.target_progress = progress
        
        # 开始新的进度动画
        self.progress_animation.setStartValue(self._progress_value)
        self.progress_animation.setEndValue(self.target_progress)
        self.progress_animation.start()
    
    def _call_finished_callback(self):
        """调用完成回调函数"""
        if hasattr(self, 'on_finished_callback') and self.on_finished_callback:
            self.on_finished_callback()
            self.on_finished_callback = None  # 防止重复调用
        self.hide()  # 确保隐藏启动屏幕
    
    def set_progress(self, value):
        """直接设置进度条值
        
        Args:
            value: 进度值(0-100)
        """
        if 0 <= value <= 100:
            # 停止当前动画
            if self.progress_animation.state() == QPropertyAnimation.Running:
                self.progress_animation.stop()
            
            # 设置新的动画
            self.progress_animation.setStartValue(self.current_progress)
            self.progress_animation.setEndValue(value)
            self.progress_animation.start()
            
            # 更新目标进度
            self.target_progress = value
            
            # 如果设置为100%，则完成动画
            if value == 100:
                self.progress_animation.finished.connect(self.finish_animation)
    
    def finish_animation(self):
        """完成动画后执行的方法，用于隐藏启动屏幕并启动主窗口"""
        # 更新消息
        self.message_label.setText("加载完成!")
        QApplication.processEvents()
        
        # 短暂延迟后调用回调
        if hasattr(self, 'on_finished_callback') and self.on_finished_callback:
            QTimer.singleShot(800, self._call_finished_callback)
        else:
            # 如果没有回调，只是隐藏启动屏幕
            QTimer.singleShot(800, self.hide)
    
    def mousePressEvent(self, event):
        """点击事件处理，允许用户点击跳过启动画面"""
        # 当用户点击时立即完成动画
        self.set_progress(100) 