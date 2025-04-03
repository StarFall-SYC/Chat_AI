"""
标签页模块
包含聊天标签页和训练标签页的实现
"""

from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout,
                           QPushButton, QFrame, QSpacerItem, QSizePolicy, QMessageBox,
                           QTableWidgetItem, QListWidget, QProgressBar, QSpinBox, 
                           QDoubleSpinBox, QFileDialog, QMenu, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, QPoint
from PyQt5.QtGui import QFont, QTextCursor, QPixmap, QImage, QTextDocument
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os
import sys
import json
import traceback
from typing import List, Dict, Any, Optional
import time

# 确保models目录在sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入自定义组件
from ui.custom_widgets import ChatMessageList, AnimatedButton

# 检查matplotlib是否可用
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 尝试导入模型相关模块
try:
    from models.chatbot import ChatbotManager
    from models import app_logger
except ImportError:
    # 定义一个简单的ChatbotManager替代品
    class ChatbotManager:
        def __init__(self, model_path=None):
            pass
        def get_response(self, text, include_media=True):
            return "模型未加载，无法回复", 0.0, None, None

    # 创建日志记录器占位符
    class DummyLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    app_logger = DummyLogger()

class ChatHistoryManager:
    """对话历史管理器，用于保存和加载对话历史"""
    
    def __init__(self, max_history=100):
        """初始化
        
        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.history = []
        self.current_session = []
    
    def add_message(self, role, text, media_path=None, media_type="text"):
        """添加消息到当前会话"""
        message = {
            "role": role,
            "text": text,
            "media_path": media_path,
            "media_type": media_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.current_session.append(message)
        
        # 如果超过最大历史记录数量，删除最早的记录
        if len(self.current_session) > self.max_history:
            self.current_session.pop(0)
    
    def save_session(self, session_name=None):
        """保存当前会话"""
        if not session_name:
            session_name = f"会话_{time.strftime('%Y%m%d_%H%M%S')}"
        
        session = {
            "name": session_name,
            "messages": self.current_session.copy(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.history.append(session)
        return session
    
    def clear_session(self):
        """清除当前会话"""
        self.current_session = []
    
    def get_current_session(self):
        """获取当前会话"""
        return self.current_session
    
    def get_session_by_name(self, name):
        """通过名称获取会话"""
        for session in self.history:
            if session["name"] == name:
                return session
        return None
    
    def get_all_sessions(self):
        """获取所有会话"""
        return self.history
    
    def save_to_file(self, file_path):
        """保存所有历史记录到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "history": self.history,
                    "current_session": self.current_session
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存历史记录失败: {str(e)}")
            return False
    
    def load_from_file(self, file_path):
        """从文件加载历史记录"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.history = data.get("history", [])
                self.current_session = data.get("current_session", [])
            return True
        except Exception as e:
            print(f"加载历史记录失败: {str(e)}")
            return False

class ChatTab(QWidget):
    """聊天标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # 使用多模态模型
        try:
            from models.advanced_model import MultiModalModel
            self.multi_modal_model = MultiModalModel()
            self.use_advanced_model = True
        except ImportError:
            # 如果高级模型不可用，回退到基础模型
            self.chatbot = ChatbotManager()
            self.use_advanced_model = False
            
        # 初始化对话历史管理器
        self.history_manager = ChatHistoryManager()
        
        # 存储用户偏好
        self.user_preferences = {
            "auto_scroll": True,
            "typing_effect": True,
            "voice_enabled": True,
            "theme": "light",
            "font_size": "medium"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建状态标签（暂时不显示，但保留以兼容旧代码）
        self.status_label = QLabel()
        self.status_label.hide()
        
        # 创建聊天消息列表
        self.message_list = ChatMessageList()
        self.message_list.setObjectName("chatMessageList")
        main_layout.addWidget(self.message_list, 1)  # 1是拉伸因子
        
        # 创建输入区域容器
        input_container = QWidget()
        input_container.setObjectName("inputArea")
        input_container.setStyleSheet("""
            QWidget#inputArea {
                background-color: #f5f5f5;
                border-top: 1px solid #e0e0e0;
                min-height: 100px;
                max-height: 180px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(20, 10, 20, 10)
        input_layout.setSpacing(5)
        
        # 创建输入框和按钮的水平布局
        input_row_layout = QHBoxLayout()
        input_row_layout.setContentsMargins(0, 0, 0, 0)
        input_row_layout.setSpacing(10)
        
        # 创建更美观的输入框
        self.input_box = QTextEdit()
        self.input_box.setObjectName("chatInput")
        self.input_box.setPlaceholderText("在此输入消息...")
        self.input_box.setMinimumHeight(60)
        self.input_box.setMaximumHeight(100)
        self.input_box.setStyleSheet("""
            QTextEdit#chatInput {
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                background-color: white;
                padding: 8px;
                font-size: 10pt;
            }
            QTextEdit#chatInput:focus {
                border: 1px solid #3498db;
            }
        """)
        
        # 设置按键事件
        self.input_box.keyPressEvent = self.input_key_press
        input_row_layout.addWidget(self.input_box)
        
        # 创建按钮布局
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        
        # 添加多媒体上传按钮
        self.media_button = QPushButton("多媒体")
        self.media_button.setObjectName("mediaButton")
        self.media_button.setMinimumWidth(80)
        self.media_button.setMinimumHeight(30)
        self.media_button.setCursor(Qt.PointingHandCursor)
        self.media_button.setStyleSheet("""
            QPushButton#mediaButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton#mediaButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton#mediaButton:pressed {
                background-color: #6A1B9A;
            }
        """)
        self.media_button.clicked.connect(self.show_media_menu)
        button_layout.addWidget(self.media_button)
        
        # 创建更现代的发送按钮
        send_button = QPushButton("发送")
        send_button.setObjectName("sendButton")
        send_button.setMinimumWidth(80)
        send_button.setMinimumHeight(30)
        send_button.setCursor(Qt.PointingHandCursor)
        send_button.setStyleSheet("""
            QPushButton#sendButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton#sendButton:hover {
                background-color: #1976D2;
            }
            QPushButton#sendButton:pressed {
                background-color: #0D47A1;
            }
        """)
        send_button.clicked.connect(self.send_message)
        button_layout.addWidget(send_button)
        
        # 将按钮布局添加到输入行布局
        input_row_layout.addLayout(button_layout)
        
        # 添加输入行到主输入布局
        input_layout.addLayout(input_row_layout)
        
        # 创建底部提示行
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 0, 0)
        
        # 添加更优雅的提示文本
        hint_label = QLabel("按 Enter 发送  •  按 Shift+Enter 换行")
        hint_label.setStyleSheet("""
            color: #888888; 
            font-size: 9pt;
            background-color: transparent;
            padding: 2px 0;
        """)
        bottom_layout.addWidget(hint_label)
        
        # 添加弹性空间
        bottom_layout.addStretch(1)
        
        # 添加历史会话按钮
        history_button = QPushButton("历史会话")
        history_button.setStyleSheet("""
            background-color: transparent;
            color: #2196F3;
            border: none;
            font-size: 9pt;
            text-decoration: underline;
        """)
        history_button.setCursor(Qt.PointingHandCursor)
        history_button.clicked.connect(self.show_history_dialog)
        bottom_layout.addWidget(history_button)
        
        # 添加底部布局到主输入布局
        input_layout.addLayout(bottom_layout)
        
        # 将输入容器添加到主布局
        main_layout.addWidget(input_container)
        
        # 初始状态下显示欢迎消息
        self.add_message("assistant", "您好，我是AI助手，有什么可以帮助您的吗？")
    
    def input_key_press(self, event):
        """处理输入框的按键事件"""
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.input_box, event)
        
    def send_message(self):
        """发送消息"""
        # 获取输入文本并清空输入框
        message = self.input_box.toPlainText().strip()
        self.input_box.clear()
        
        if not message:
            return
        
        # 显示用户消息
        self.add_message("user", message)
        
        # 保存到历史管理器
        self.history_manager.add_message("user", message)
        
        # 显示加载动画
        self.message_list.start_loading_animation()
        
        # 设置定时器延迟回复，模拟思考时间
        QTimer.singleShot(800, lambda: self.generate_ai_response(message))
    
    def generate_ai_response(self, user_message):
        """生成AI响应（模拟）"""
        # 这里我们模拟智能回复，实际应用中会连接到真实的模型
        
        # 基于用户输入创建模拟回答
        responses = {
            "你好": ["您好！有什么可以帮助您的吗？", "您好！今天有什么需要我协助的吗？", "您好！我是您的AI助手，请问有什么需要帮助的？"],
            "介绍": ["我是一个基于先进语言模型的AI助手，可以回答问题、提供信息、进行对话等。", 
                  "我是您的AI助手，可以帮助您解决问题、获取信息，以及进行日常对话。",
                  "我是一个智能对话机器人，擅长自然语言处理和对话生成。"],
            "功能": ["我可以回答问题、提供信息、生成内容、协助学习等，但目前仍在不断进步中。",
                  "我的主要功能包括问答、对话、内容生成、知识查询等，不过我还在学习更多技能。",
                  "我能够理解自然语言，进行对话交流，提供信息和建议，但我的能力仍有局限性。"],
            "谢谢": ["不客气！随时为您服务。", "很高兴能帮到您！", "不用谢，有其他问题随时问我。"],
            "再见": ["再见！祝您有愉快的一天！", "下次再见！", "再见，期待下次为您服务！"]
        }
        
        # 检查上下文，实现简单的对话连贯性
        session = self.history_manager.get_current_session()
        if len(session) >= 4:  # 至少有2轮对话
            last_bot_message = None
            for msg in reversed(session[:-1]):  # 排除最后一条用户消息
                if msg["role"] == "assistant":
                    last_bot_message = msg["text"]
                    break
            
            # 上一轮我问过用户问题，检查是否回答了
            if last_bot_message and ("您能" in last_bot_message or "能否" in last_bot_message or "请问" in last_bot_message):
                # 用户可能在回答问题
                response_text = f"谢谢您的回复。根据您提到的'{user_message[:20]}...'，我理解您的意思了。在实际应用中，我会根据上下文提供更连贯的回答。"
                self.type_ai_response(response_text)
                self.history_manager.add_message("assistant", response_text)
                return
        
        # 智能匹配响应
        response_text = ""
        for key, replies in responses.items():
            if key in user_message.lower():
                import random
                response_text = random.choice(replies)
                break
        
        # 如果没有匹配到预设回复，生成一个通用回复
        if not response_text:
            if "什么" in user_message or "如何" in user_message or "怎么" in user_message or "?" in user_message or "？" in user_message:
                response_text = f"关于'{user_message}'，这是一个很好的问题。在实际应用中，我会连接到知识库为您提供详细的解答。目前我只能告诉您，这个问题涉及到的内容比较专业，需要根据特定情况具体分析。您可以尝试提供更多细节，以便我更好地理解您的需求。"
            elif len(user_message) < 10:
                response_text = "您的消息很简短，能否详细说明一下您的需求，这样我才能更好地为您提供帮助。"
            else:
                response_text = f"我理解您提到的是关于'{user_message[:15]}...'的内容。在实际应用中，我会根据您的输入生成更有针对性的回答。目前我正在学习理解和回应各种类型的问题，感谢您的耐心使用。"
        
        # 随机添加后续问题，增强交互性
        import random
        if random.random() < 0.3:  # 30%的概率添加后续问题
            follow_up_questions = [
                "您还有其他问题吗？",
                "有什么地方我可以为您进一步解释的吗？",
                "需要了解更多相关信息吗？",
                "您对这个回答满意吗？"
            ]
            response_text += f"\n\n{random.choice(follow_up_questions)}"
        
        # 模拟打字效果，逐字显示回复
        self.type_ai_response(response_text)
        
        # 保存到历史管理器
        self.history_manager.add_message("assistant", response_text)
    
    def type_ai_response(self, text, current_pos=0):
        """模拟AI打字效果"""
        # 停止加载动画
        if current_pos == 0:
            self.message_list.stop_loading_animation()
            self.current_bubble = self.add_message("assistant", "")
        
        # 如果已经完成打字，返回
        if current_pos >= len(text):
            return
        
        # 更新当前显示的文本
        current_text = text[:current_pos+1]
        self.current_bubble.text_display.setPlainText(current_text)
        self.current_bubble.text = current_text
        
        # 计算下一个字符的延迟（模拟真实打字速度）
        import random
        delay = random.randint(50, 100)  # 50-100毫秒
        
        # 在标点符号处稍作停顿
        if current_pos < len(text) - 1 and text[current_pos] in "，。！？,.!?":
            delay += random.randint(200, 400)  # 额外增加200-400毫秒
        
        # 段落切换处增加更长的停顿
        if current_pos < len(text) - 1 and text[current_pos] == '\n':
            delay += random.randint(400, 800)
        
        # 设置定时器显示下一个字符
        QTimer.singleShot(delay, lambda: self.type_ai_response(text, current_pos + 1))
    
    def add_message(self, role, text, media_path=None, media_type="text"):
        """添加消息到对话框
        
        Args:
            role: 角色(user/assistant)
            text: 消息文本
            media_path: 媒体文件路径
            media_type: 媒体类型(text/image/video)
        """
        return self.message_list.add_message(role, text, media_path, media_type)
    
    def clear_chat(self):
        """清除聊天历史"""
        # 弹出确认对话框
        reply = QMessageBox.question(
            self, "确认清除", 
            "确定要清除所有聊天记录吗？这个操作无法撤销。",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 保存当前会话到历史记录
            if self.message_list.message_count > 1:  # 如果不只有欢迎消息
                self.history_manager.save_session()
            
            # 清除消息列表
            self.message_list.clear_messages()
            
            # 清除当前会话
            self.history_manager.clear_session()
            
            # 添加欢迎消息
            welcome_message = "聊天已清除，有什么可以帮助您的吗？"
            self.add_message("assistant", welcome_message)
            self.history_manager.add_message("assistant", welcome_message)
    
    def show_media_menu(self, position=None):
        """显示多媒体上传菜单"""
        # 创建菜单
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 5px;
            }
            QMenu::item {
                padding: 6px 25px 6px 25px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #e3f2fd;
                color: #1565C0;
            }
        """)
        
        # 添加上传图片菜单项
        upload_image = QAction("上传图片", self)
        upload_image.triggered.connect(self.upload_image)
        menu.addAction(upload_image)
        
        # 添加上传视频菜单项
        upload_video = QAction("上传视频", self)
        upload_video.triggered.connect(self.upload_video)
        menu.addAction(upload_video)
        
        # 添加上传音频菜单项
        upload_audio = QAction("上传音频", self)
        upload_audio.triggered.connect(self.upload_audio)
        menu.addAction(upload_audio)
        
        # 添加上传文档菜单项
        upload_document = QAction("上传文档", self)
        upload_document.triggered.connect(self.upload_document)
        menu.addAction(upload_document)
        
        # 显示菜单
        if position:
            menu.exec_(self.mapToGlobal(position))
        else:
            menu.exec_(self.media_button.mapToGlobal(
                QPoint(0, self.media_button.height())))
            
    def upload_image(self):
        """上传图片（模拟）"""
        # 打开文件对话框选择图片
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        
        if not file_path:
            return
            
        # 显示图片消息
        self.add_message("user", "我上传了一张图片：", file_path, "image")
        
        # 保存到历史管理器
        self.history_manager.add_message("user", "我上传了一张图片", file_path, "image")
        
        # 显示加载动画
        self.message_list.start_loading_animation()
        
        # 模拟AI分析图片并回复
        QTimer.singleShot(1500, lambda: self.analyze_image(file_path))
    
    def upload_video(self):
        """上传视频（模拟）"""
        # 打开文件对话框选择视频
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if not file_path:
            return
            
        # 显示视频消息
        self.add_message("user", "我上传了一段视频：", file_path, "video")
        
        # 保存到历史管理器
        self.history_manager.add_message("user", "我上传了一段视频", file_path, "video")
        
        # 显示加载动画
        self.message_list.start_loading_animation()
        
        # 模拟AI分析视频并回复
        QTimer.singleShot(1500, lambda: self.analyze_video(file_path))
    
    def upload_audio(self):
        """上传音频（模拟）"""
        # 打开文件对话框选择音频
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "", "音频文件 (*.mp3 *.wav *.ogg *.m4a)"
        )
        
        if not file_path:
            return
            
        # 显示音频消息（当前UI不支持音频播放，仅作为文本显示）
        self.add_message("user", f"我上传了一段音频：{os.path.basename(file_path)}")
        
        # 保存到历史管理器
        self.history_manager.add_message("user", f"我上传了一段音频：{os.path.basename(file_path)}")
        
        # 显示加载动画
        self.message_list.start_loading_animation()
        
        # 模拟AI分析音频并回复
        QTimer.singleShot(1500, lambda: self.analyze_audio(file_path))
    
    def upload_document(self):
        """上传文档（模拟）"""
        # 打开文件对话框选择文档
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文档", "", "文档文件 (*.pdf *.doc *.docx *.txt *.md)"
        )
        
        if not file_path:
            return
            
        # 显示文档消息
        self.add_message("user", f"我上传了一份文档：{os.path.basename(file_path)}")
        
        # 保存到历史管理器
        self.history_manager.add_message("user", f"我上传了一份文档：{os.path.basename(file_path)}")
        
        # 显示加载动画
        self.message_list.start_loading_animation()
        
        # 模拟AI分析文档并回复
        QTimer.singleShot(1500, lambda: self.analyze_document(file_path))
    
    def analyze_image(self, image_path):
        """分析图片（模拟）"""
        # 在实际应用中，这里会调用图像识别API
        # 这里我们简单模拟一些分析结果
        import random
        import os
        
        # 获取文件名作为提示
        file_name = os.path.basename(image_path)
        
        # 模拟识别结果
        analysis_templates = [
            f"我看到了这张图片，文件名为{file_name}。这是一张很清晰的图片，我能识别到图片中的内容。在实际应用中，我会通过图像识别算法提取图片中的关键信息和主体内容。",
            f"感谢您上传的图片。图片文件{file_name}已成功加载。从图片分析来看，这似乎是一张高质量的图片。在真实的应用场景中，我会提供更详细的图像分析结果。",
            f"我已接收到您的图片{file_name}。我能识别这是一张图片文件，但由于当前演示环境的限制，无法进行实际的图像分析。在完整版本中，我可以识别图片中的物体、人物、场景等元素。"
        ]
        
        analysis_text = random.choice(analysis_templates)
        
        # 附加一些场景特定的回复
        if "cat" in file_name.lower() or "猫" in file_name:
            analysis_text += "\n\n看起来这可能是一张猫的图片。猫是很受欢迎的宠物，已经与人类共同生活了数千年。"
        elif "dog" in file_name.lower() or "狗" in file_name:
            analysis_text += "\n\n这可能是一张狗的图片。狗是人类最忠实的朋友，有着丰富的品种多样性。"
        elif "food" in file_name.lower() or "食物" in file_name or "菜" in file_name:
            analysis_text += "\n\n这看起来像是一张食物的图片。食物图片是社交媒体上最常分享的内容之一。"
            
        # 模拟打字效果显示分析结果
        self.type_ai_response(analysis_text)
        
        # 保存到历史管理器
        self.history_manager.add_message("assistant", analysis_text)
    
    def analyze_video(self, video_path):
        """分析视频（模拟）"""
        import random
        import os
        
        # 获取文件名作为提示
        file_name = os.path.basename(video_path)
        
        # 模拟识别结果
        analysis_templates = [
            f"我收到了您上传的视频 {file_name}。在实际应用中，我会通过视频分析技术对视频内容进行识别和理解。目前我可以确认这是一个视频文件，您可以在聊天界面中查看和播放。",
            f"您的视频 {file_name} 已成功上传。这是一个视频文件，在完整版应用中，我能够分析视频中的场景、动作和物体，提取关键帧信息，甚至识别视频中的文字和语音。",
            f"我已接收到您分享的视频 {file_name}。在实际应用场景中，我会对视频内容进行全面分析，包括场景分类、物体跟踪和活动识别等。"
        ]
        
        analysis_text = random.choice(analysis_templates)
        
        # 附加一些视频特定的回复
        file_name_lower = file_name.lower()
        if "talk" in file_name_lower or "speech" in file_name_lower or "演讲" in file_name:
            analysis_text += "\n\n这似乎是一段演讲或讲话视频。在完整版中，我可以提取视频中的语音内容并转换为文本。"
        elif "nature" in file_name_lower or "landscape" in file_name_lower or "风景" in file_name:
            analysis_text += "\n\n这看起来可能是一段风景视频。大自然的美景总是能给人带来放松和愉悦的感受。"
        elif "tutorial" in file_name_lower or "guide" in file_name_lower or "教程" in file_name:
            analysis_text += "\n\n这可能是一段教程视频。教程视频是分享知识和技能的有效方式。"
            
        # 模拟打字效果显示分析结果
        self.type_ai_response(analysis_text)
        
        # 保存到历史管理器
        self.history_manager.add_message("assistant", analysis_text)
    
    def analyze_audio(self, audio_path):
        """分析音频（模拟）"""
        import random
        import os
        
        # 获取文件名作为提示
        file_name = os.path.basename(audio_path)
        
        # 模拟识别结果
        analysis_templates = [
            f"我已接收到您上传的音频文件 {file_name}。在实际应用中，我会通过语音识别技术将音频转换为文本，并进行进一步分析。",
            f"您的音频文件 {file_name} 已上传成功。当前界面不支持音频播放，但在完整版中，我能够识别语音内容，提取关键信息，甚至识别说话者身份。",
            f"我收到了您分享的音频 {file_name}。在实际应用场景中，我会分析音频内容，包括语音转文字、情感分析和关键词提取等。"
        ]
        
        analysis_text = random.choice(analysis_templates)
        
        # 附加一些音频特定的回复
        file_name_lower = file_name.lower()
        if "music" in file_name_lower or "song" in file_name_lower or "音乐" in file_name or "歌" in file_name:
            analysis_text += "\n\n这似乎是一个音乐文件。在完整版中，我可以识别音乐类型、节奏和可能的歌曲名称。"
        elif "interview" in file_name_lower or "talk" in file_name_lower or "访谈" in file_name or "对话" in file_name:
            analysis_text += "\n\n这可能是一段访谈或对话录音。对话分析可以帮助提取重要信息和观点。"
        elif "lecture" in file_name_lower or "speech" in file_name_lower or "讲座" in file_name or "演讲" in file_name:
            analysis_text += "\n\n这看起来可能是一段讲座或演讲录音。演讲内容通常包含丰富的知识和见解。"
            
        # 模拟打字效果显示分析结果
        self.type_ai_response(analysis_text)
        
        # 保存到历史管理器
        self.history_manager.add_message("assistant", analysis_text)
    
    def analyze_document(self, document_path):
        """分析文档（模拟）"""
        import random
        import os
        
        # 获取文件名和扩展名
        file_name = os.path.basename(document_path)
        file_extension = os.path.splitext(document_path)[1].lower()
        
        # 模拟识别结果
        analysis_templates = [
            f"我已接收到您上传的文档 {file_name}。在实际应用中，我会提取文档内容并进行分析，帮助您理解文档的主要内容和关键点。",
            f"您的文档 {file_name} 已上传成功。在完整版应用中，我能够解析文档结构，提取文本内容，甚至识别表格和图表信息。",
            f"我收到了您分享的文档 {file_name}。在实际应用场景中，我会分析文档内容，包括主题识别、关键信息提取和摘要生成等。"
        ]
        
        analysis_text = random.choice(analysis_templates)
        
        # 根据文件类型添加特定回复
        if file_extension == ".pdf":
            analysis_text += "\n\nPDF文件通常包含格式化文本、图像、表格等元素。在完整版中，我能够提取所有这些内容并进行结构化分析。"
        elif file_extension in [".doc", ".docx"]:
            analysis_text += "\n\nWord文档通常用于创建结构化的文本内容。在完整版中，我能够识别文档的标题、段落、列表等结构元素。"
        elif file_extension in [".txt", ".md"]:
            analysis_text += "\n\n这是一个纯文本文件，通常包含未格式化的文本内容。这种格式简单直接，适合快速记录信息。"
            
        # 模拟文档内容分析
        document_types = ["报告", "论文", "备忘录", "教程", "新闻稿", "合同", "简历"]
        random_type = random.choice(document_types)
        
        analysis_text += f"\n\n根据文件名分析，这可能是一份{random_type}文档。在实际应用中，我会对文档内容进行更深入的理解和分析。"
        
        # 模拟打字效果显示分析结果
        self.type_ai_response(analysis_text)
        
        # 保存到历史管理器
        self.history_manager.add_message("assistant", analysis_text)
    
    def show_history_dialog(self):
        """显示历史会话对话框"""
        # 这里只是一个简单的模拟实现
        # 实际应用中需要创建一个适当的对话框组件
        
        # 保存当前会话
        if self.message_list.message_count > 1:
            session_name = f"会话_{time.strftime('%Y%m%d_%H%M%S')}"
            self.history_manager.save_session(session_name)
        
        # 获取所有历史会话
        sessions = self.history_manager.get_all_sessions()
        
        if not sessions:
            QMessageBox.information(self, "历史会话", "暂无历史会话记录")
            return
            
        # 创建一个简单的消息框显示历史会话
        history_text = "历史会话列表：\n\n"
        for i, session in enumerate(sessions):
            message_count = len(session["messages"])
            history_text += f"{i+1}. {session['name']} ({session['timestamp']})\n"
            history_text += f"   包含 {message_count} 条消息\n\n"
        
        QMessageBox.information(self, "历史会话", history_text)

class DataLoader:
    """数据加载器，用于处理不同类型的训练数据文件"""
    
    @staticmethod
    def load_from_directory(directory_path):
        """从目录中加载所有支持的数据文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            list: 训练数据列表
        """
        training_data = []
        
        # 统计信息
        stats = {
            "json_files": 0,
            "text_files": 0,
            "image_files": 0,
            "video_files": 0,
            "other_files": 0,
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0
        }
        
        # 遍历目录中的所有文件
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                stats["total_files"] += 1
                
                try:
                    # 根据文件扩展名处理不同类型的文件
                    if file.lower().endswith('.json'):
                        stats["json_files"] += 1
                        json_data = DataLoader.load_json_file(file_path)
                        if json_data:
                            training_data.extend(json_data)
                            stats["processed_files"] += 1
                        
                    elif file.lower().endswith(('.txt', '.md')):
                        stats["text_files"] += 1
                        text_data = DataLoader.load_text_file(file_path)
                        if text_data:
                            training_data.append(text_data)
                            stats["processed_files"] += 1
                            
                    elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        stats["image_files"] += 1
                        image_data = DataLoader.load_image_file(file_path)
                        if image_data:
                            training_data.append(image_data)
                            stats["processed_files"] += 1
                            
                    elif file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        stats["video_files"] += 1
                        video_data = DataLoader.load_video_file(file_path)
                        if video_data:
                            training_data.append(video_data)
                            stats["processed_files"] += 1
                            
                    else:
                        stats["other_files"] += 1
                        
                except Exception as e:
                    stats["failed_files"] += 1
                    print(f"处理文件 {file_path} 出错: {str(e)}")
        
        return training_data, stats
    
    @staticmethod
    def load_json_file(file_path):
        """加载JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            list: 训练数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持两种格式：直接的意图列表或者包含在"intents"键中
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "intents" in data:
                return data["intents"]
            else:
                return []
        except Exception as e:
            print(f"加载JSON文件 {file_path} 出错: {str(e)}")
            return []
    
    @staticmethod
    def load_text_file(file_path):
        """加载文本文件，自动创建训练数据条目
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 训练数据条目
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # 使用文件名作为标签
            file_name = os.path.basename(file_path)
            tag = os.path.splitext(file_name)[0].lower().replace(' ', '_')
            
            # 将文本内容分割成段落，每个段落作为一个模式
            patterns = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # 如果段落太长，则进一步分割
            processed_patterns = []
            for p in patterns:
                if len(p) > 200:  # 如果段落超过200个字符
                    sentences = [s.strip() for s in p.split('.') if s.strip()]
                    processed_patterns.extend(sentences)
                else:
                    processed_patterns.append(p)
            
            # 过滤过长的句子
            filtered_patterns = [p for p in processed_patterns if len(p) < 500]
            
            # 生成一些回复
            responses = [
                f"这是关于{tag}的信息。",
                f"以下是{tag}的相关内容。",
                f"关于{tag}，我知道这些信息。"
            ]
            
            return {
                "tag": tag,
                "patterns": filtered_patterns[:30],  # 限制样本数量
                "responses": responses,
                "source": file_path
            }
        except Exception as e:
            print(f"加载文本文件 {file_path} 出错: {str(e)}")
            return None
    
    @staticmethod
    def load_image_file(file_path):
        """加载图像文件，创建基于图像的训练数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 训练数据条目
        """
        try:
            # 使用文件名作为标签
            file_name = os.path.basename(file_path)
            tag = os.path.splitext(file_name)[0].lower().replace(' ', '_') + "_image"
            
            # 创建与图像相关的模式
            patterns = [
                f"显示 {tag} 图片",
                f"我想看 {tag}",
                f"展示 {tag} 的图像",
                f"给我看 {tag}",
                f"{tag} 是什么样的",
                f"你能展示 {tag} 吗"
            ]
            
            # 创建回复
            responses = [
                f"这是 {tag} 的图片。",
                f"这是 {tag} 的图像。",
                f"这是关于 {tag} 的视觉内容。"
            ]
            
            return {
                "tag": tag,
                "patterns": patterns,
                "responses": responses,
                "media_type": "image",
                "media_path": file_path,
                "source": file_path
            }
        except Exception as e:
            print(f"加载图像文件 {file_path} 出错: {str(e)}")
            return None
    
    @staticmethod
    def load_video_file(file_path):
        """加载视频文件，创建基于视频的训练数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 训练数据条目
        """
        try:
            # 使用文件名作为标签
            file_name = os.path.basename(file_path)
            tag = os.path.splitext(file_name)[0].lower().replace(' ', '_') + "_video"
            
            # 创建与视频相关的模式
            patterns = [
                f"播放 {tag} 视频",
                f"我想看 {tag} 视频",
                f"展示 {tag} 的视频",
                f"给我看 {tag} 的录像",
                f"{tag} 视频是什么样的",
                f"你能播放 {tag} 视频吗"
            ]
            
            # 创建回复
            responses = [
                f"这是 {tag} 的视频。",
                f"这是 {tag} 的录像。",
                f"这是关于 {tag} 的视频内容。"
            ]
            
            return {
                "tag": tag,
                "patterns": patterns,
                "responses": responses,
                "media_type": "video",
                "media_path": file_path,
                "source": file_path
            }
        except Exception as e:
            print(f"加载视频文件 {file_path} 出错: {str(e)}")
            return None

class TrainingTab(QWidget):
    """训练标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chatbot = ChatbotManager()
        self.training_data = []
        self.training_thread = None
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        
        # 训练数据加载区域
        data_group_layout = QVBoxLayout()
        data_group_layout.addWidget(QLabel("训练数据"))
        
        # 数据源选择区
        data_source_layout = QHBoxLayout()
        
        # 添加数据加载按钮
        self.load_dir_button = QPushButton("从文件夹加载")
        self.load_dir_button.clicked.connect(self.load_from_directory)
        data_source_layout.addWidget(self.load_dir_button)
        
        self.load_file_button = QPushButton("从JSON文件加载")
        self.load_file_button.clicked.connect(self.load_from_json)
        data_source_layout.addWidget(self.load_file_button)
        
        # 数据统计信息
        self.data_stats_label = QLabel("未加载训练数据")
        data_source_layout.addWidget(self.data_stats_label, 1)
        
        data_group_layout.addLayout(data_source_layout)
        
        # 数据预览列表
        self.data_preview = QListWidget()
        self.data_preview.setMaximumHeight(150)
        data_group_layout.addWidget(self.data_preview)
        
        layout.addLayout(data_group_layout)
        
        # 训练设置区域
        settings_layout = QHBoxLayout()
        
        # 训练参数设置
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(200)
        settings_layout.addWidget(QLabel("训练轮数:"))
        settings_layout.addWidget(self.epochs_spin)
        
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setSingleStep(0.0001)
        settings_layout.addWidget(QLabel("学习率:"))
        settings_layout.addWidget(self.learning_rate_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        settings_layout.addWidget(QLabel("批大小:"))
        settings_layout.addWidget(self.batch_size_spin)
        
        layout.addLayout(settings_layout)
        
        # 训练控制按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        # 添加保存训练结果按钮
        self.save_button = QPushButton("保存模型")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 训练曲线
        if MATPLOTLIB_AVAILABLE:
            self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        
        # 状态文本
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
        
    def load_from_directory(self):
        """从目录加载训练数据"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择训练数据目录", os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not directory:
            return
            
        self.update_status(f"正在从目录 {directory} 加载训练数据...")
        
        # 在后台线程中加载数据
        self.load_thread = DataLoadingThread(directory)
        self.load_thread.loading_completed.connect(self.on_data_loaded)
        self.load_thread.loading_failed.connect(self.on_data_loading_failed)
        self.load_thread.start()
        
        # 禁用加载按钮
        self.load_dir_button.setEnabled(False)
        self.load_file_button.setEnabled(False)
    
    def load_from_json(self):
        """从JSON文件加载训练数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据JSON文件", os.path.expanduser("~"),
            "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
            
        self.update_status(f"正在从文件 {file_path} 加载训练数据...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持两种格式：直接的意图列表或者包含在"intents"键中
            if isinstance(data, list):
                self.training_data = data
            elif isinstance(data, dict) and "intents" in data:
                self.training_data = data["intents"]
            else:
                self.update_status("错误: JSON文件格式不正确")
                return
                
            self.update_data_preview()
            self.update_status(f"成功加载 {len(self.training_data)} 个训练意图")
            
            # 更新统计信息
            self.data_stats_label.setText(f"已加载: {len(self.training_data)} 个意图")
            
        except Exception as e:
            self.update_status(f"加载数据出错: {str(e)}")
    
    def on_data_loaded(self, data, stats):
        """数据加载完成的回调
        
        Args:
            data: 加载的训练数据
            stats: 数据统计信息
        """
        self.training_data = data
        self.update_data_preview()
        
        # 构建统计信息字符串
        stats_text = f"已加载: {len(data)} 个意图 ({stats['processed_files']}/{stats['total_files']} 个文件)"
        self.data_stats_label.setText(stats_text)
        
        # 显示详细统计信息
        stats_detail = (
            f"已处理文件: {stats['processed_files']}/{stats['total_files']}\n"
            f"JSON文件: {stats['json_files']}\n"
            f"文本文件: {stats['text_files']}\n"
            f"图像文件: {stats['image_files']}\n"
            f"视频文件: {stats['video_files']}\n"
            f"其他文件: {stats['other_files']}\n"
            f"失败文件: {stats['failed_files']}"
        )
        
        self.update_status(f"数据加载完成!\n{stats_detail}")
        
        # 重新启用加载按钮
        self.load_dir_button.setEnabled(True)
        self.load_file_button.setEnabled(True)
    
    def on_data_loading_failed(self, error_message):
        """数据加载失败的回调"""
        self.update_status(f"数据加载失败: {error_message}")
        
        # 重新启用加载按钮
        self.load_dir_button.setEnabled(True)
        self.load_file_button.setEnabled(True)
    
    def update_data_preview(self):
        """更新数据预览列表"""
        self.data_preview.clear()
        
        for i, intent in enumerate(self.training_data[:100]):  # 限制显示前100个
            tag = intent.get("tag", f"未命名_{i}")
            patterns_count = len(intent.get("patterns", []))
            
            # 检查是否有媒体
            media_type = intent.get("media_type", "")
            media_info = f" [{media_type}]" if media_type else ""
            
            item_text = f"{tag}: {patterns_count} 模式{media_info}"
            self.data_preview.addItem(item_text)
        
    def start_training(self):
        """开始训练"""
        if not self.training_data:
            QMessageBox.warning(self, "警告", "没有训练数据，请先加载数据")
            return
            
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练已经在进行中")
            return
            
        # 创建训练线程
        self.training_thread = TrainingWorker(
            self.chatbot,
            self.training_data,
            epochs=self.epochs_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            batch_size=self.batch_size_spin.value()
        )
        
        # 连接信号
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.status_updated.connect(self.update_status)
        self.training_thread.epoch_completed.connect(self.on_epoch_completed)
        self.training_thread.training_completed.connect(self.on_training_completed)
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()
        
        # 开始训练
        self.training_thread.start()
        
    def stop_training(self):
        """停止训练"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认停止", 
                "确定要停止当前训练吗？这可能导致模型不完整。",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.update_status("正在停止训练...")
        else:
            QMessageBox.information(self, "提示", "没有正在进行的训练")
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """更新状态文本"""
        self.status_text.append(message)
        # 滚动到底部
        self.status_text.moveCursor(QTextCursor.End)
        
    def on_epoch_completed(self, epoch, loss, accuracy):
        """每轮训练完成的处理"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        if MATPLOTLIB_AVAILABLE:
            self.update_training_curves()
            
    def on_training_completed(self, success, message):
        """训练完成的处理"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            self.save_button.setEnabled(True)  # 启用保存按钮
            QMessageBox.information(self, "训练完成", "模型训练已成功完成！")
        else:
            self.progress_bar.setValue(0)
            self.save_button.setEnabled(False)  # 禁用保存按钮
            QMessageBox.critical(self, "训练失败", f"训练失败：{message}")
            
        if self.training_thread:
            self.training_thread.wait()
            self.training_thread.deleteLater()
            self.training_thread = None
            
    def update_training_curves(self):
        """更新训练曲线"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(self.epochs, self.losses, 'r-')
        self.ax1.set_title('训练损失')
        self.ax1.set_xlabel('轮次')
        self.ax1.set_ylabel('损失值')
        self.ax1.grid(True)
        
        self.ax2.plot(self.epochs, self.accuracies, 'b-')
        self.ax2.set_title('训练准确率')
        self.ax2.set_xlabel('轮次')
        self.ax2.set_ylabel('准确率')
        self.ax2.set_ylim([0, 1.05])
        self.ax2.grid(True)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def save_model(self):
        """保存训练好的模型到用户指定位置"""
        # 获取当前模型路径
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_path, 'data', 'models')
        source_model_path = os.path.join(models_dir, 'chat_model.pth')
        
        # 检查模型是否存在
        if not os.path.exists(source_model_path):
            QMessageBox.warning(self, "错误", "找不到已训练的模型文件")
            return
        
        # 选择保存路径
        target_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存模型文件", 
            os.path.expanduser("~"), 
            "模型文件 (*.pth)"
        )
        
        if not target_path:
            return
            
        try:
            # 复制模型文件
            import shutil
            shutil.copy2(source_model_path, target_path)
            
            # 尝试同时保存训练数据
            if self.training_data:
                # 获取不带扩展名的文件路径
                base_target_path = os.path.splitext(target_path)[0]
                data_path = f"{base_target_path}_training_data.json"
                
                # 保存训练数据
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_data, f, ensure_ascii=False, indent=2)
                
                self.update_status(f"模型已保存到: {target_path}\n训练数据已保存到: {data_path}")
                QMessageBox.information(self, "保存成功", f"模型和训练数据已成功保存！")
            else:
                self.update_status(f"模型已保存到: {target_path}")
                QMessageBox.information(self, "保存成功", "模型已成功保存！")
                
        except Exception as e:
            self.update_status(f"保存模型时出错: {str(e)}")
            QMessageBox.critical(self, "保存失败", f"保存模型时出错: {str(e)}")
            traceback.print_exc()

class DataLoadingThread(QThread):
    """数据加载线程"""
    loading_completed = pyqtSignal(list, dict)  # 发送加载的数据和统计信息
    loading_failed = pyqtSignal(str)  # 发送错误消息
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        
    def run(self):
        """运行线程"""
        try:
            # 从目录加载数据
            data, stats = DataLoader.load_from_directory(self.directory)
            
            # 发出加载完成信号
            self.loading_completed.emit(data, stats)
            
        except Exception as e:
            # 发出加载失败信号
            self.loading_failed.emit(str(e))
            traceback.print_exc()

class TrainingWorker(QThread):
    """训练工作线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    epoch_completed = pyqtSignal(int, float, float)
    training_completed = pyqtSignal(bool, str)
    
    def __init__(self, chatbot, training_data, epochs=200, learning_rate=0.001, batch_size=32):
        super().__init__()
        self.chatbot = chatbot
        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_running = True
        
    def stop(self):
        """停止训练"""
        self.is_running = False
        
    def run(self):
        """执行训练过程"""
        try:
            self.status_updated.emit("开始训练模型...")
            self.progress_updated.emit(0)
            
            # 训练模型
            success = self.chatbot.train(
                self.training_data,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size
            )
            
            if success:
                self.training_completed.emit(True, "训练成功完成！")
            else:
                self.training_completed.emit(False, "训练过程异常终止")
                
        except Exception as e:
            self.status_updated.emit(f"训练出错: {str(e)}")
            self.status_updated.emit(traceback.format_exc())
            self.training_completed.emit(False, str(e)) 