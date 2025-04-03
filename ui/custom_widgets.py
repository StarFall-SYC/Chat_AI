"""
自定义控件模块
包含高级UI组件和自定义小部件
"""

from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout,
                           QPushButton, QFrame, QScrollArea, QSizePolicy, QSpacerItem,
                           QProgressBar, QToolButton, QMenu, QAction, QApplication, 
                           QDialog, QFileDialog, QMessageBox, QToolTip, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QSize, QRect, QPropertyAnimation, QEasingCurve, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QFont, QColor, QPalette, QPainterPath, QIcon, QCursor
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import os

class ChatBubble(QFrame):
    """聊天气泡组件"""
    
    def __init__(self, text, role="user", parent=None):
        """初始化聊天气泡
        
        Args:
            text: 消息文本
            role: 角色(user/assistant)
            parent: 父组件
        """
        super().__init__(parent)
        
        self.role = role
        self.text = text
        self.media_path = None
        self.media_type = None
        self.media_player = None
        self.video_widget = None
        
        # 设置样式
        self.setObjectName(f"{role}Bubble")
        self.setStyleSheet(self._get_bubble_style())
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 12, 15, 12)  # 减少垂直方向的外边距
        self.layout.setSpacing(6)  # 减少组件间距
        
        # 顶部布局: 角色标签和时间戳
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        
        # 添加角色标签 - 改进样式
        self.role_label = QLabel("你:" if role == "user" else "AI助手:")
        self.role_label.setObjectName("roleLabel")
        self.role_label.setStyleSheet(f"""
            font-weight: bold; 
            color: {('#1565C0' if role == 'user' else '#212121')}; 
            background: transparent;
            font-size: 10pt;
        """)
        top_layout.addWidget(self.role_label)
        
        # 添加时间戳 - 改进样式
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M")
        self.time_label = QLabel(timestamp)
        self.time_label.setObjectName("timeLabel")
        self.time_label.setStyleSheet("font-size: 9pt; color: #9e9e9e; background: transparent;")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_layout.addWidget(self.time_label, 1)  # 1是拉伸因子
        
        self.layout.addLayout(top_layout)
        
        # 添加文本内容 - 改进样式
        self.text_display = QTextEdit()
        self.text_display.setObjectName("textDisplay")
        self.text_display.setReadOnly(True)
        self.text_display.setFrameShape(QFrame.NoFrame)
        self.text_display.setPlainText(text)
        self.text_display.setStyleSheet(f"""
            border: none; 
            background-color: transparent;
            font-size: 11pt;
            line-height: 1.4;
            color: {'#1a237e' if role == 'user' else '#212121'};
            selection-background-color: {'#bbdefb' if role == 'user' else '#e0e0e0'};
        """)
        
        # 自动调整高度
        doc_height = self.text_display.document().size().toSize().height()
        self.text_display.setFixedHeight(min(max(doc_height + 10, 30), 300))
        
        # 如果内容超过最大高度，启用滚动
        if doc_height > 300:
            self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.layout.addWidget(self.text_display)
        
        # 设置右键菜单
        self.text_display.setContextMenuPolicy(Qt.CustomContextMenu)
        self.text_display.customContextMenuRequested.connect(self._show_context_menu)
        
        # 设置尺寸策略
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    
    def _show_context_menu(self, position):
        """显示右键菜单"""
        context_menu = QMenu()
        copy_action = QAction("复制文本", self)
        copy_action.triggered.connect(self._copy_text)
        context_menu.addAction(copy_action)
        
        # 如果有媒体，添加媒体相关选项
        if self.media_path:
            if self.media_type == "image":
                view_action = QAction("查看原图", self)
                view_action.triggered.connect(lambda: self._view_full_image(self.media_path))
                context_menu.addAction(view_action)
            
            save_action = QAction("保存" + ("图片" if self.media_type == "image" else "视频"), self)
            save_action.triggered.connect(lambda: self._save_media(self.media_type, self.media_path))
            context_menu.addAction(save_action)
        
        context_menu.exec_(self.text_display.mapToGlobal(position))
    
    def _copy_text(self):
        """复制文本到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text)
        
        # 显示临时提示消息
        QApplication.instance().setOverrideCursor(Qt.ArrowCursor)
        QToolTip.showText(QCursor.pos(), "已复制到剪贴板", self, QRect(), 1500)
    
    def _get_bubble_style(self):
        """获取气泡样式"""
        if self.role == "user":
            return """
                QFrame {
                    background-color: #e3f2fd;
                    border-radius: 16px;
                    border-top-right-radius: 4px;
                    border: 1px solid #bbdefb;
                }
            """
        else:
            return """
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 16px;
                    border-top-left-radius: 4px;
                    border: 1px solid #e0e0e0;
                }
            """
    
    def add_media(self, media_path, media_type):
        """添加媒体内容
        
        Args:
            media_path: 媒体文件路径
            media_type: 媒体类型(image/video)
        """
        if not os.path.exists(media_path):
            error_label = QLabel(f"媒体文件不存在: {media_path}")
            error_label.setStyleSheet("color: red; padding: 8px; background-color: #ffebee; border-radius: 4px;")
            error_label.setWordWrap(True)
            self.layout.addWidget(error_label)
            return
            
        self.media_path = media_path
        self.media_type = media_type
        
        # 媒体容器
        media_container = QFrame()
        media_container.setObjectName("mediaContainer")
        media_container.setStyleSheet("background-color: rgba(0,0,0,0.03); border-radius: 8px; padding: 4px;")
        media_layout = QVBoxLayout(media_container)
        media_layout.setContentsMargins(8, 8, 8, 8)
        media_layout.setSpacing(6)
        
        # 媒体标题
        media_title = QLabel("图像" if media_type == "image" else "视频")
        media_title.setStyleSheet("font-weight: bold; font-size: 10pt; color: #555555;")
        media_layout.addWidget(media_title)
        
        if media_type == "image":
            # 创建图像预览
            image_widget = QLabel()
            pixmap = QPixmap(media_path)
            
            # 保持纵横比的情况下调整大小
            max_width = 450  # 最大宽度
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
            
            image_widget.setPixmap(pixmap)
            image_widget.setAlignment(Qt.AlignCenter)
            media_layout.addWidget(image_widget)
            
            # 添加图像工具栏
            img_toolbar = QHBoxLayout()
            img_toolbar.setContentsMargins(0, 0, 0, 0)
            img_toolbar.setSpacing(8)
            
            # 添加查看原图按钮
            view_button = QPushButton("查看原图")
            view_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            view_button.clicked.connect(lambda: self._view_full_image(media_path))
            img_toolbar.addWidget(view_button)
            
            # 添加保存按钮
            save_button = QPushButton("保存图像")
            save_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            save_button.clicked.connect(lambda: self._save_media("image", media_path))
            img_toolbar.addWidget(save_button)
            
            img_toolbar.addStretch(1)
            
            # 添加文件信息
            file_info = os.path.basename(media_path)
            info_label = QLabel(file_info)
            info_label.setStyleSheet("font-size: 9pt; color: #888888;")
            img_toolbar.addWidget(info_label)
            
            media_layout.addLayout(img_toolbar)
            
        elif media_type == "video":
            # 创建视频容器
            video_container = QWidget()
            video_layout = QVBoxLayout(video_container)
            video_layout.setContentsMargins(0, 0, 0, 0)
            video_layout.setSpacing(4)
            
            # 创建视频预览
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(240)
            self.video_widget.setMinimumWidth(320)
            self.video_widget.setStyleSheet("background-color: #000000;")
            video_layout.addWidget(self.video_widget)
            
            # 创建视频控制栏
            control_layout = QHBoxLayout()
            control_layout.setContentsMargins(0, 0, 0, 0)
            control_layout.setSpacing(4)
            
            # 播放按钮
            play_button = QPushButton("播放")
            play_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            play_button.clicked.connect(self._play_video)
            control_layout.addWidget(play_button)
            
            # 暂停按钮
            pause_button = QPushButton("暂停")
            pause_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            pause_button.clicked.connect(self._pause_video)
            control_layout.addWidget(pause_button)
            
            # 停止按钮
            stop_button = QPushButton("停止")
            stop_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            stop_button.clicked.connect(self._stop_video)
            control_layout.addWidget(stop_button)
            
            # 添加保存按钮
            save_button = QPushButton("保存视频")
            save_button.setStyleSheet("font-size: 9pt; padding: 4px 8px;")
            save_button.clicked.connect(lambda: self._save_media("video", media_path))
            control_layout.addWidget(save_button)
            
            # 添加弹性空间
            control_layout.addStretch(1)
            
            # 添加视频信息
            file_info = os.path.basename(media_path)
            info_label = QLabel(file_info)
            info_label.setStyleSheet("font-size: 9pt; color: #888888;")
            control_layout.addWidget(info_label)
            
            video_layout.addLayout(control_layout)
            
            # 初始化媒体播放器
            try:
                from PyQt5.QtCore import QUrl
                self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
                self.media_player.setVideoOutput(self.video_widget)
                self.media_player.setMedia(QUrl.fromLocalFile(media_path))
                
                # 获取视频缩略图，如果可用
                video_thumbnail = QLabel()
                video_thumbnail.setAlignment(Qt.AlignCenter)
                video_thumbnail.setMinimumHeight(30)
                video_thumbnail.setText("视频准备就绪")
                video_thumbnail.setStyleSheet("color: #555555; font-style: italic;")
                video_layout.addWidget(video_thumbnail)
                
            except Exception as e:
                error_label = QLabel(f"无法加载视频: {str(e)}")
                error_label.setStyleSheet("color: red; background-color: #ffebee; padding: 4px; border-radius: 4px;")
                video_layout.addWidget(error_label)
            
            media_layout.addWidget(video_container)
        
        # 添加媒体容器到主布局
        self.layout.addWidget(media_container)
    
    def _view_full_image(self, image_path):
        """查看原始大小的图像"""
        dialog = QDialog()
        dialog.setWindowTitle("查看图像")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # 创建滚动区域以允许查看大图
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 创建图像标签
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(QPixmap(image_path))
        
        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)
        
        # 创建关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec_()
    
    def _save_media(self, media_type, source_path):
        """保存媒体文件"""
        file_type = "图像" if media_type == "image" else "视频"
        extension = os.path.splitext(source_path)[1]
        
        # 获取保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            f"保存{file_type}", 
            os.path.expanduser(f"~/{os.path.basename(source_path)}"),
            f"{file_type}文件 (*{extension})"
        )
        
        if not save_path:
            return
            
        try:
            # 复制文件
            import shutil
            shutil.copy2(source_path, save_path)
            QMessageBox.information(self, "成功", f"{file_type}已保存到: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存{file_type}时出错: {str(e)}")
    
    def _play_video(self):
        """播放视频"""
        if self.media_player:
            if self.media_player.state() != QMediaPlayer.PlayingState:
                self.media_player.play()
    
    def _pause_video(self):
        """暂停视频"""
        if self.media_player and self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
    
    def _stop_video(self):
        """停止视频"""
        if self.media_player:
            self.media_player.stop()

class ChatMessageList(QScrollArea):
    """聊天消息列表"""
    
    def __init__(self, parent=None):
        """初始化聊天消息列表"""
        super().__init__(parent)
        
        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        
        # 创建内容容器
        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        self.container.setStyleSheet("""
            QWidget#chatContainer {
                background-color: transparent;
            }
        """)
        self.setWidget(self.container)
        
        # 使用垂直布局
        self.message_layout = QVBoxLayout(self.container)
        self.message_layout.setAlignment(Qt.AlignTop)
        self.message_layout.setSpacing(16)
        self.message_layout.setContentsMargins(16, 16, 16, 16)
        
        # 添加底部的弹性空间
        self.spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.message_layout.addItem(self.spacer)
        
        # 消息计数器
        self.message_count = 0
        
        # 创建新消息提示按钮(当滚动到上方时显示)
        self.new_message_button = QPushButton("↓ 新消息")
        self.new_message_button.setObjectName("newMessageButton")
        self.new_message_button.setStyleSheet("""
            QPushButton#newMessageButton {
                background-color: #2196F3;
                color: white;
                border-radius: 15px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton#newMessageButton:hover {
                background-color: #1E88E5;
            }
        """)
        self.new_message_button.setCursor(Qt.PointingHandCursor)
        self.new_message_button.clicked.connect(self.scroll_to_bottom)
        self.new_message_button.setFixedHeight(30)
        self.new_message_button.setFixedWidth(100)
        self.new_message_button.setParent(self)
        self.new_message_button.hide()
        
        # 创建加载动画
        self.loading_indicator = QLabel("正在生成回复...", self)
        self.loading_indicator.setAlignment(Qt.AlignCenter)
        self.loading_indicator.setStyleSheet("""
            background-color: rgba(0,0,0,0.7);
            color: white;
            border-radius: 15px;
            padding: 8px 15px;
        """)
        self.loading_indicator.setFixedHeight(36)
        self.loading_indicator.setFixedWidth(130)
        self.loading_indicator.hide()
        
        # 设置定时器用于显示动画
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._update_loading_animation)
        self.loading_dots = 0
        
        # 滚动位置监控
        self.verticalScrollBar().valueChanged.connect(self._handle_scroll)
        self.should_auto_scroll = True
        self.new_message_added = False
    
    def resizeEvent(self, event):
        """窗口大小改变事件，用于重新定位按钮和加载指示器"""
        super().resizeEvent(event)
        
        # 更新新消息按钮位置
        button_x = (self.width() - self.new_message_button.width()) // 2
        button_y = self.height() - self.new_message_button.height() - 20
        self.new_message_button.move(button_x, button_y)
        
        # 更新加载指示器位置
        indicator_x = (self.width() - self.loading_indicator.width()) // 2
        indicator_y = self.height() - self.loading_indicator.height() - 70
        self.loading_indicator.move(indicator_x, indicator_y)
    
    def _handle_scroll(self, value):
        """处理滚动事件"""
        max_value = self.verticalScrollBar().maximum()
        
        # 如果滚动接近底部，则启用自动滚动
        if max_value - value < 50:
            self.should_auto_scroll = True
            self.new_message_button.hide()
        else:
            self.should_auto_scroll = False
            # 如果有新消息且未在底部，显示新消息按钮
            if self.new_message_added:
                self.new_message_button.show()
    
    def add_message(self, role, text, media_path=None, media_type=None):
        """添加消息
        
        Args:
            role: 角色(user/assistant)
            text: 消息文本
            media_path: 媒体文件路径，可选
            media_type: 媒体类型，可选
            
        Returns:
            ChatBubble: 创建的消息气泡
        """
        # 移除底部弹性空间
        self.message_layout.removeItem(self.spacer)
        
        # 如果是助手回复，先隐藏加载指示器
        if role == "assistant":
            self.stop_loading_animation()
        
        # 创建消息气泡
        bubble = ChatBubble(text, role)
        
        # 如果有媒体内容，添加到气泡
        if media_path and media_type:
            bubble.add_media(media_path, media_type)
        
        # 将气泡添加到布局中
        self.message_layout.addWidget(bubble)
        
        # 添加分隔线（除了第一条消息）
        if self.message_count > 0:
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            separator.setStyleSheet("background-color: rgba(0,0,0,0.05); margin: 0 30px;")
            separator.setMaximumHeight(1)
            # 将分隔线插入到新气泡前面
            self.message_layout.insertWidget(self.message_layout.count() - 1, separator)
        
        # 更新消息计数
        self.message_count += 1
        
        # 重新添加底部弹性空间
        self.message_layout.addItem(self.spacer)
        
        # 设置新消息标志
        self.new_message_added = True
        
        # 处理滚动
        QApplication.processEvents()
        if self.should_auto_scroll:
            self.scroll_to_bottom()
        else:
            # 如果不在底部，显示新消息提示
            self.new_message_button.show()
        
        return bubble
    
    def scroll_to_bottom(self):
        """滚动到底部"""
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.new_message_button.hide()
        self.new_message_added = False
    
    def clear_messages(self):
        """清空所有消息"""
        # 移除底部弹性空间
        if self.spacer in [self.message_layout.itemAt(i) for i in range(self.message_layout.count())]:
            self.message_layout.removeItem(self.spacer)
        
        # 删除所有消息气泡
        while self.message_layout.count() > 0:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 重新添加底部弹性空间
        self.message_layout.addItem(self.spacer)
        
        # 重置消息计数
        self.message_count = 0
        self.new_message_added = False
        self.new_message_button.hide()
    
    def start_loading_animation(self):
        """开始加载动画"""
        self.loading_dots = 0
        self.loading_indicator.show()
        self.loading_timer.start(500)  # 每500毫秒更新一次
        
        # 确保动画显示在正确位置
        indicator_x = (self.width() - self.loading_indicator.width()) // 2
        indicator_y = self.height() - self.loading_indicator.height() - 70
        self.loading_indicator.move(indicator_x, indicator_y)
    
    def stop_loading_animation(self):
        """停止加载动画"""
        self.loading_timer.stop()
        self.loading_indicator.hide()
    
    def _update_loading_animation(self):
        """更新加载动画文本"""
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        self.loading_indicator.setText(f"正在生成回复{dots}")
        
        # 确保动画可见
        if self.should_auto_scroll:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
            
    def get_messages_text(self):
        """获取所有消息文本
        
        Returns:
            list: 包含所有消息的列表，每条消息是一个字典
        """
        messages = []
        
        for i in range(self.message_layout.count()):
            item = self.message_layout.itemAt(i)
            widget = item.widget()
            
            # 检查是否是ChatBubble类型
            if isinstance(widget, ChatBubble):
                messages.append({
                    "role": widget.role,
                    "text": widget.text,
                    "media_path": widget.media_path,
                    "media_type": widget.media_type
                })
        
        return messages

class AnimatedButton(QPushButton):
    """带动画效果的按钮"""
    
    def __init__(self, text="", parent=None):
        """初始化按钮"""
        super().__init__(text, parent)
        
        # 设置属性
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(40)  # 确保按钮有足够的高度
        
        # 动画相关
        self._animation = QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(100)  # 较短的动画持续时间
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 记录原始尺寸
        self._original_size = None
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        if not self._original_size:
            self._original_size = self.size()
        
        # 创建放大动画
        target_rect = QRect(
            self.geometry().x() - 2,
            self.geometry().y() - 2,
            self._original_size.width() + 4,
            self._original_size.height() + 4
        )
        
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(target_rect)
        self._animation.start()
        
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        if not self._original_size:
            self._original_size = self.size()
        
        # 创建恢复动画
        target_rect = QRect(
            self.geometry().x() + 2,
            self.geometry().y() + 2,
            self._original_size.width(),
            self._original_size.height()
        )
        
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(target_rect)
        self._animation.start()
        
        super().leaveEvent(event)

class FloatingActionButton(QToolButton):
    """悬浮动作按钮"""
    
    clicked = pyqtSignal()
    
    def __init__(self, icon_path=None, parent=None):
        """初始化按钮"""
        super().__init__(parent)
        
        # 设置固定大小
        self.setFixedSize(50, 50)
        
        # 添加阴影效果（通过样式表实现）
        self.setStyleSheet("""
            QToolButton {
                background-color: #2196F3;
                color: white;
                border-radius: 25px;
                border: none;
                font-size: 18px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #1E88E5;
            }
            QToolButton:pressed {
                background-color: #1976D2;
            }
        """)
        
        # 设置图标
        if icon_path and os.path.exists(icon_path):
            self.setIcon(QIcon(icon_path))
        
        # 设置图标大小
        self.setIconSize(QSize(24, 24))
        
        # 设置鼠标样式
        self.setCursor(Qt.PointingHandCursor)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # 连接点击信号
        self.clicked.connect(self.on_clicked)
    
    def setText(self, text):
        """设置按钮文本"""
        super().setText(text)
        # 如果文本不为空，则调整图标位置
        if text:
            self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    
    def on_clicked(self):
        """点击处理"""
        # 添加点击效果动画
        anim = QPropertyAnimation(self, b"pos")
        anim.setDuration(100)
        anim.setStartValue(self.pos())
        anim.setEndValue(self.pos())
        anim.setEasingCurve(QEasingCurve.OutBounce)
        anim.start(QPropertyAnimation.DeleteWhenStopped)
        
        # 发送点击信号
        self.clicked.emit()

class RoundProgressBar(QProgressBar):
    """圆形进度条"""
    
    def __init__(self, parent=None):
        """初始化进度条"""
        super().__init__(parent)
        
        # 设置固定大小
        self.setFixedSize(100, 100)
        
        # 设置无框架
        self.setTextVisible(False)
        self.setFrameShape(QFrame.NoFrame)
        
        # 自定义属性
        self.line_width = 8
        self.inner_radius = (self.width() - self.line_width) // 2 - 10
        
        # 颜色
        self.bg_color = QColor("#E0E0E0")
        self.progress_color = QColor("#2196F3")
        
        # 设置样式
        self.setStyleSheet("""
            QProgressBar {
                background-color: transparent;
                border: none;
            }
            
            QProgressBar::chunk {
                background-color: transparent;
            }
        """)
    
    def paintEvent(self, event):
        """重写绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 圆心和半径
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # 计算进度角度
        progress = self.value()
        max_value = self.maximum()
        progress_angle = (progress / max_value) * 360
        
        # 绘制背景圆环
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.bg_color)
        painter.drawEllipse(center_x - self.inner_radius, center_y - self.inner_radius, 
                         self.inner_radius * 2, self.inner_radius * 2)
        
        # 绘制进度圆弧
        if progress > 0:
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.progress_color)
            
            # 创建圆弧路径
            path = QPainterPath()
            path.moveTo(center_x, center_y)
            path.arcTo(center_x - self.inner_radius, center_y - self.inner_radius,
                     self.inner_radius * 2, self.inner_radius * 2,
                     90, -progress_angle)
            path.lineTo(center_x, center_y)
            
            painter.drawPath(path)
        
        # 绘制中心圆
        painter.setBrush(QColor("white"))
        painter.drawEllipse(center_x - self.inner_radius + self.line_width, 
                         center_y - self.inner_radius + self.line_width,
                         (self.inner_radius - self.line_width) * 2,
                         (self.inner_radius - self.line_width) * 2)
        
        # 绘制文本
        painter.setPen(QColor("#333333"))
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        text = f"{progress}%"
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        
        painter.end()

class ModernButton(QPushButton):
    """现代化按钮组件，提供美观的圆角渐变效果"""
    
    def __init__(self, text="", parent=None, primary=True):
        """初始化
        
        Args:
            text: 按钮文本
            parent: 父组件
            primary: 是否为主要按钮(主要按钮使用主题色，次要按钮使用灰色)
        """
        super().__init__(text, parent)
        self.primary = primary
        self._animation = None
        self._hovered = False
        self._pressed = False
        
        # 设置样式
        self.setCursor(Qt.PointingHandCursor)
        
        # 基础样式
        self.setMinimumHeight(36)
        self.setFont(QFont("Microsoft YaHei", 9))
        
        # 根据类型应用不同样式
        self.update_style()
    
    def update_style(self):
        """更新样式"""
        if self.primary:
            self.setProperty("class", "")
        else:
            self.setProperty("class", "secondary")
        
        # 确保样式表被应用
        self.style().unpolish(self)
        self.style().polish(self)
    
    def setPrimary(self, primary):
        """设置是否为主要按钮"""
        if self.primary != primary:
            self.primary = primary
            self.update_style()
    
    def setIcon(self, icon):
        """设置图标"""
        super().setIcon(icon)
        # 图标的尺寸
        self.setIconSize(QSize(16, 16))
        
    def sizeHint(self):
        """获取默认尺寸"""
        size = super().sizeHint()
        size.setHeight(max(size.height(), 36))
        size.setWidth(max(size.width(), 80))
        return size
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        self._hovered = True
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self._hovered = False
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self._pressed = True
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self._pressed = False
        super().mouseReleaseEvent(event)


class ModernToolButton(QToolButton):
    """现代化工具按钮，提供美观的图标式按钮"""
    
    def __init__(self, parent=None):
        """初始化"""
        super().__init__(parent)
        self._animation = None
        self._hovered = False
        self._pressed = False
        
        # 设置样式
        self.setCursor(Qt.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # 基础样式
        self.setMinimumSize(64, 64)
        self.setFont(QFont("Microsoft YaHei", 9))
        self.setIconSize(QSize(32, 32))
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        self._hovered = True
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self._hovered = False
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self._pressed = True
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self._pressed = False
        super().mouseReleaseEvent(event)


class StyledCard(QFrame):
    """卡片式容器，具有阴影效果"""
    
    def __init__(self, parent=None):
        """初始化"""
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setMidLineWidth(0)
        
        # 设置内部布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
    
    def addWidget(self, widget):
        """添加子组件"""
        self.layout.addWidget(widget)
    
    def addLayout(self, layout):
        """添加子布局"""
        self.layout.addLayout(layout)


class StatusInfoWidget(QWidget):
    """状态信息组件，用于显示各种类型的状态消息"""
    
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    
    def __init__(self, text="", message_type=INFO, parent=None):
        """初始化
        
        Args:
            text: 显示的文本
            message_type: 消息类型(info, success, warning, error)
            parent: 父组件
        """
        super().__init__(parent)
        self.message_type = message_type
        
        # 创建布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # 添加图标
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(16, 16)
        layout.addWidget(self.icon_label)
        
        # 添加文本
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)
        
        # 设置样式
        self.setObjectName("status_info")
        self.set_type(message_type)
    
    def set_type(self, message_type):
        """设置消息类型"""
        self.message_type = message_type
        
        # 设置适当的CSS类
        self.setProperty("class", f"{message_type}-message")
        
        # 确保样式表被应用
        self.style().unpolish(self)
        self.style().polish(self)
    
    def setText(self, text):
        """设置文本"""
        self.text_label.setText(text) 