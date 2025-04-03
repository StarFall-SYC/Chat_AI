"""
高级功能标签页模块
包含多模态内容生成、情感分析等高级功能的标签页
"""

import os
import sys
import time
import threading
import traceback
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QTextEdit, QLabel, QProgressBar, QTabWidget, 
                           QFileDialog, QScrollArea, QFrame, QSlider,
                           QSpinBox, QDoubleSpinBox, QComboBox, QGridLayout,
                           QCheckBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QMutex, QWaitCondition
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

# 确保models目录在sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.advanced_model import ImageGenerator, VideoGenerator, MultiModalModel
    from models.lock_manager import lock_manager, ResourceLock
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

# 已废弃的全局资源锁，保留以供向后兼容
# 但实际使用新的lock_manager
IMAGE_GENERATOR_LOCK = threading.RLock()
VIDEO_GENERATOR_LOCK = threading.RLock()
MODEL_LOCKS = {}  # 模型ID -> 锁的映射

def get_model_lock(model_id):
    """获取指定模型ID的锁，如果不存在则创建"""
    # 此函数保留以供向后兼容，但实际使用lock_manager
    if model_id not in MODEL_LOCKS:
        MODEL_LOCKS[model_id] = threading.RLock()
    return MODEL_LOCKS[model_id]

class ImageGenerationWorker(QThread):
    """图像生成工作线程"""
    # 定义信号
    progress_updated = pyqtSignal(str)
    generation_complete = pyqtSignal(str)  # 成功时传递图像路径
    generation_failed = pyqtSignal(str)    # 失败时传递错误信息
    
    def __init__(self, prompt, negative_prompt=None, width=512, height=512, model_id="runwayml/stable-diffusion-v1-5"):
        """初始化
        
        Args:
            prompt: 生成提示
            negative_prompt: 负面提示
            width: 宽度
            height: 高度
            model_id: 模型ID
        """
        super().__init__()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.model_id = model_id
        self.generator = None
        self.running = True
        # 不再直接创建模型锁，使用锁管理器
    
    def run(self):
        """运行线程"""
        # 使用锁管理器获取图像生成器锁
        try:
            with ResourceLock("image_generator", "global", timeout=1):
                self.progress_updated.emit("正在初始化图像生成模型...")
                
                # 创建生成器实例
                self.generator = ImageGenerator(model_id=self.model_id)
                
                # 使用模型锁
                try:
                    with ResourceLock(self.model_id, "model", timeout=2):
                        # 初始化生成器
                        if not self.generator.initialized:
                            success = self.generator.initialize()
                            if not success:
                                self.generation_failed.emit("初始化图像生成模型失败")
                                return
                        
                        self.progress_updated.emit("开始生成图像...")
                        
                        # 生成图像
                        start_time = time.time()
                        image = self.generator.generate(
                            prompt=self.prompt,
                            negative_prompt=self.negative_prompt,
                            width=self.width,
                            height=self.height
                        )
                except TimeoutError:
                    self.generation_failed.emit(f"无法获取模型 {self.model_id} 资源，可能有其他任务正在使用该模型")
                    return
                
                if not image:
                    self.generation_failed.emit("图像生成失败")
                    return
                    
                # 添加水印
                image = self.generator.add_watermark(image)
                
                # 确保生成的图像不会有文件名冲突
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unique_id = os.urandom(4).hex()  # 添加随机ID防止文件名冲突
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                save_dir = os.path.join(base_path, 'data', 'generated', 'images')
                
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    self.generation_failed.emit(f"创建保存目录失败: {str(e)}")
                    return
                    
                save_path = os.path.join(save_dir, f"generated_image_{timestamp}_{unique_id}.png")
                
                try:
                    image.save(save_path)
                except Exception as e:
                    self.generation_failed.emit(f"保存图像失败: {str(e)}")
                    return
                    
                elapsed = time.time() - start_time
                self.progress_updated.emit(f"图像生成完成，耗时 {elapsed:.2f} 秒")
                
                # 完成信号
                self.generation_complete.emit(save_path)
                
        except TimeoutError:
            self.generation_failed.emit("无法获取图像生成器资源，可能有其他生成任务正在进行")
            return
        except Exception as e:
            traceback.print_exc()
            self.generation_failed.emit(f"生成过程中出错: {str(e)}")
    
    def stop(self):
        """停止线程"""
        self.running = False

class VideoGenerationWorker(QThread):
    """视频生成工作线程"""
    # 定义信号
    progress_updated = pyqtSignal(str)
    generation_complete = pyqtSignal(str)  # 成功时传递视频路径
    generation_failed = pyqtSignal(str)    # 失败时传递错误信息
    
    def __init__(self, prompt, frame_count=5, fps=24, model_id="runwayml/stable-diffusion-v1-5"):
        """初始化
        
        Args:
            prompt: 生成提示
            frame_count: 帧数
            fps: 帧率
            model_id: 图像生成模型ID
        """
        super().__init__()
        self.prompt = prompt
        self.frame_count = frame_count
        self.fps = fps
        self.model_id = model_id
        self.generator = None
        self.running = True
        # 不再直接创建模型锁，使用锁管理器
    
    def run(self):
        """运行线程"""
        # 使用锁管理器同时获取视频生成器锁和图像生成器锁
        try:
            # 先获取视频生成器锁
            with ResourceLock("video_generator", "global", timeout=1):
                # 再获取图像生成器锁
                try:
                    with ResourceLock("image_generator", "global", timeout=1):
                        self.progress_updated.emit("正在初始化视频生成模型...")
                        
                        # 创建生成器实例
                        self.generator = VideoGenerator()
                        
                        # 使用模型锁
                        try:
                            with ResourceLock(self.model_id, "model", timeout=2):
                                # 初始化图像生成器
                                if not self.generator.image_generator.initialized:
                                    success = self.generator.image_generator.initialize()
                                    if not success:
                                        self.generation_failed.emit("初始化图像生成模型失败")
                                        return
                                
                                self.progress_updated.emit(f"开始生成视频，将生成 {self.frame_count} 帧...")
                                
                                # 确保生成的视频不会有文件名冲突
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                unique_id = os.urandom(4).hex()  # 添加随机ID防止文件名冲突
                                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                save_dir = os.path.join(base_path, 'data', 'generated', 'videos')
                                
                                try:
                                    os.makedirs(save_dir, exist_ok=True)
                                except Exception as e:
                                    self.generation_failed.emit(f"创建保存目录失败: {str(e)}")
                                    return
                                
                                save_path = os.path.join(save_dir, f"generated_video_{timestamp}_{unique_id}.mp4")
                                
                                # 生成视频
                                start_time = time.time()
                                
                                # 生成每一帧
                                frames = []
                                for i in range(self.frame_count):
                                    if not self.running:
                                        self.generation_failed.emit("视频生成被用户取消")
                                        return
                                        
                                    self.progress_updated.emit(f"正在生成第 {i+1}/{self.frame_count} 帧...")
                                    
                                    # 使用随机种子确保每帧不同
                                    seed = int(time.time() * 1000) % 10000 + i * 100
                                    torch_seed = None
                                    np_seed = None
                                    
                                    try:
                                        import torch
                                        import numpy as np
                                        torch_seed = torch.initial_seed()
                                        np_seed = np.random.get_state()
                                        torch.manual_seed(seed)
                                        np.random.seed(seed)
                                    except ImportError:
                                        pass
                                    
                                    # 生成帧
                                    frame = self.generator.image_generator.generate(
                                        prompt=f"{self.prompt}, frame {i+1} of {self.frame_count}",
                                        width=512,
                                        height=512
                                    )
                                    
                                    # 恢复种子
                                    if torch_seed is not None and np_seed is not None:
                                        try:
                                            torch.manual_seed(torch_seed)
                                            np.random.set_state(np_seed)
                                        except:
                                            pass
                                    
                                    if frame:
                                        frames.append(frame)
                                    else:
                                        self.generation_failed.emit(f"第 {i+1} 帧生成失败")
                                        return
                                
                                if not self.running:
                                    self.generation_failed.emit("视频生成被用户取消")
                                    return
                        except TimeoutError:
                            self.generation_failed.emit(f"无法获取模型 {self.model_id} 资源，可能有其他任务正在使用该模型")
                            return
                                    
                        # 从帧生成视频
                        self.progress_updated.emit("正在合成视频...")
                        video_path = self.generator.generate_from_frames(frames, save_path, self.fps)
                        
                        if not video_path:
                            self.generation_failed.emit("视频合成失败")
                            return
                            
                        elapsed = time.time() - start_time
                        self.progress_updated.emit(f"视频生成完成，耗时 {elapsed:.2f} 秒")
                        
                        # 完成信号
                        self.generation_complete.emit(video_path)
                except TimeoutError:
                    self.generation_failed.emit("无法获取图像生成器资源，可能有其他图像生成任务正在进行")
                    return
        except TimeoutError:
            self.generation_failed.emit("无法获取视频生成器资源，可能有其他生成任务正在进行")
            return
        except Exception as e:
            traceback.print_exc()
            self.generation_failed.emit(f"生成过程中出错: {str(e)}")
    
    def stop(self):
        """停止线程"""
        self.running = False

class ImageGenerationTab(QWidget):
    """图像生成标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.generator_thread = None
        self.is_generating = False
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 参数设置区域
        params_group = QGroupBox("生成参数")
        params_layout = QGridLayout()
        
        # 提示词
        params_layout.addWidget(QLabel("提示词:"), 0, 0)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("请输入详细的图像描述，例如：一只可爱的猫咪在阳光下玩耍")
        self.prompt_input.setMaximumHeight(100)
        params_layout.addWidget(self.prompt_input, 0, 1, 1, 3)
        
        # 负面提示词
        params_layout.addWidget(QLabel("负面提示词:"), 1, 0)
        self.negative_prompt_input = QTextEdit()
        self.negative_prompt_input.setPlaceholderText("请输入不希望出现在图像中的元素，例如：模糊，变形，低质量")
        self.negative_prompt_input.setMaximumHeight(50)
        params_layout.addWidget(self.negative_prompt_input, 1, 1, 1, 3)
        
        # 尺寸设置
        params_layout.addWidget(QLabel("宽度:"), 2, 0)
        self.width_input = QSpinBox()
        self.width_input.setRange(256, 1024)
        self.width_input.setValue(512)
        self.width_input.setSingleStep(64)
        params_layout.addWidget(self.width_input, 2, 1)
        
        params_layout.addWidget(QLabel("高度:"), 2, 2)
        self.height_input = QSpinBox()
        self.height_input.setRange(256, 1024)
        self.height_input.setValue(512)
        self.height_input.setSingleStep(64)
        params_layout.addWidget(self.height_input, 2, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("生成图像")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self.generate_image)
        control_layout.addWidget(self.generate_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        control_layout.addWidget(self.cancel_button)
        
        layout.addLayout(control_layout)
        
        # 进度和状态
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        # 预览区域
        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout()
        
        self.image_label = QLabel("还未生成图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        preview_layout.addWidget(self.image_label)
        
        # 保存按钮
        self.save_button = QPushButton("另存为...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        preview_layout.addWidget(self.save_button)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 保存最近生成的图像路径
        self.current_image_path = None
    
    def generate_image(self):
        """生成图像"""
        # 防止多次点击生成按钮
        if self.is_generating:
            QMessageBox.warning(self, "警告", "已有生成任务正在进行中，请等待完成或取消当前任务")
            return
            
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self.status_label.setText("错误: 提示词不能为空")
            return
        
        # 获取参数
        negative_prompt = self.negative_prompt_input.toPlainText().strip() or None
        width = self.width_input.value()
        height = self.height_input.value()
        
        # 禁用控件
        self.is_generating = True
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_button.setEnabled(False)
        
        # 创建生成线程
        self.generator_thread = ImageGenerationWorker(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height
        )
        
        # 连接信号
        self.generator_thread.progress_updated.connect(self.update_status)
        self.generator_thread.generation_complete.connect(self.on_generation_complete)
        self.generator_thread.generation_failed.connect(self.on_generation_failed)
        self.generator_thread.finished.connect(self.on_thread_finished)
        
        # 启动线程
        self.generator_thread.start()
    
    def cancel_generation(self):
        """取消生成"""
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.stop()
            self.status_label.setText("正在取消图像生成...")
            # 不要在这里重新启用按钮，等待线程实际结束后再启用
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
    
    def on_generation_complete(self, image_path):
        """生成完成回调"""
        self.current_image_path = image_path
        
        # 显示图像
        pixmap = QPixmap(image_path)
        if pixmap.width() > self.image_label.width():
            pixmap = pixmap.scaledToWidth(self.image_label.width(), Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        
        # 更新UI状态 - 注意在on_thread_finished中也会启用按钮
        self.save_button.setEnabled(True)
        
        self.status_label.setText(f"图像生成成功，保存到: {image_path}")
    
    def on_generation_failed(self, error_message):
        """生成失败回调"""
        self.status_label.setText(f"错误: {error_message}")
        
        # UI状态在on_thread_finished中更新
    
    def on_thread_finished(self):
        """线程结束回调"""
        self.is_generating = False
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # 清理线程对象
        if self.generator_thread:
            self.generator_thread.deleteLater()
            self.generator_thread = None
    
    def save_image(self):
        """保存图像到自定义路径"""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.status_label.setText("错误: 没有可保存的图像")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            os.path.expanduser("~/Pictures"),
            "图像文件 (*.png *.jpg *.jpeg)"
        )
        
        if not file_path:
            return
        
        try:
            # 加载源图像
            image = QImage(self.current_image_path)
            
            # 保存到选定路径
            saved = image.save(file_path)
            
            if saved:
                self.status_label.setText(f"图像已保存到: {file_path}")
            else:
                self.status_label.setText("保存图像失败，请检查路径是否可写")
        except Exception as e:
            self.status_label.setText(f"保存图像时出错: {str(e)}")

class VideoGenerationTab(QWidget):
    """视频生成标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.generator_thread = None
        self.is_generating = False
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 参数设置区域
        params_group = QGroupBox("生成参数")
        params_layout = QGridLayout()
        
        # 提示词
        params_layout.addWidget(QLabel("提示词:"), 0, 0)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("请输入详细的视频描述，例如：一片森林的雾气在晨曦中流动")
        self.prompt_input.setMaximumHeight(100)
        params_layout.addWidget(self.prompt_input, 0, 1, 1, 3)
        
        # 帧数设置
        params_layout.addWidget(QLabel("帧数:"), 1, 0)
        self.frames_input = QSpinBox()
        self.frames_input.setRange(2, 30)
        self.frames_input.setValue(5)
        params_layout.addWidget(self.frames_input, 1, 1)
        
        # 帧率设置
        params_layout.addWidget(QLabel("帧率:"), 1, 2)
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 30)
        self.fps_input.setValue(10)
        params_layout.addWidget(self.fps_input, 1, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("生成视频")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self.generate_video)
        control_layout.addWidget(self.generate_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        control_layout.addWidget(self.cancel_button)
        
        layout.addLayout(control_layout)
        
        # 进度和状态
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        # 预览区域
        preview_group = QGroupBox("视频预览")
        preview_layout = QVBoxLayout()
        
        # 视频播放器
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(400)
        preview_layout.addWidget(self.video_widget)
        
        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        
        # 播放控制
        playback_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.media_player.play)
        playback_layout.addWidget(self.play_button)
        
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.media_player.pause)
        playback_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.media_player.stop)
        playback_layout.addWidget(self.stop_button)
        
        preview_layout.addLayout(playback_layout)
        
        # 保存按钮
        self.save_button = QPushButton("另存为...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_video)
        preview_layout.addWidget(self.save_button)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 保存最近生成的视频路径
        self.current_video_path = None
    
    def generate_video(self):
        """生成视频"""
        # 防止多次点击生成按钮
        if self.is_generating:
            QMessageBox.warning(self, "警告", "已有生成任务正在进行中，请等待完成或取消当前任务")
            return
            
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self.status_label.setText("错误: 提示词不能为空")
            return
        
        # 获取参数
        frames = self.frames_input.value()
        fps = self.fps_input.value()
        
        # 禁用控件
        self.is_generating = True
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # 创建生成线程
        self.generator_thread = VideoGenerationWorker(
            prompt=prompt,
            frame_count=frames,
            fps=fps
        )
        
        # 连接信号
        self.generator_thread.progress_updated.connect(self.update_status)
        self.generator_thread.generation_complete.connect(self.on_generation_complete)
        self.generator_thread.generation_failed.connect(self.on_generation_failed)
        self.generator_thread.finished.connect(self.on_thread_finished)
        
        # 启动线程
        self.generator_thread.start()
    
    def cancel_generation(self):
        """取消生成"""
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.stop()
            self.status_label.setText("正在取消视频生成...")
            # 不要在这里重新启用按钮，等待线程实际结束后再启用
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
    
    def on_generation_complete(self, video_path):
        """生成完成回调"""
        self.current_video_path = video_path
        
        # 设置视频播放器
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        
        # 更新UI状态
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        self.status_label.setText(f"视频生成成功，保存到: {video_path}")
        
        # 自动播放
        self.media_player.play()
    
    def on_generation_failed(self, error_message):
        """生成失败回调"""
        self.status_label.setText(f"错误: {error_message}")
        
        # UI状态在on_thread_finished中更新
    
    def on_thread_finished(self):
        """线程结束回调"""
        self.is_generating = False
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # 清理线程对象
        if self.generator_thread:
            self.generator_thread.deleteLater()
            self.generator_thread = None
    
    def save_video(self):
        """保存视频到自定义路径"""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            self.status_label.setText("错误: 没有可保存的视频")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存视频",
            os.path.expanduser("~/Videos"),
            "视频文件 (*.mp4)"
        )
        
        if not file_path:
            return
        
        try:
            # 复制视频文件
            import shutil
            shutil.copy2(self.current_video_path, file_path)
            
            self.status_label.setText(f"视频已保存到: {file_path}")
        except Exception as e:
            self.status_label.setText(f"保存视频时出错: {str(e)}")

class MediaGenerationTab(QWidget):
    """多模态生成标签页，包含图像和视频生成"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 添加图像生成标签页
        self.image_tab = ImageGenerationTab()
        self.tab_widget.addTab(self.image_tab, "图像生成")
        
        # 添加视频生成标签页
        self.video_tab = VideoGenerationTab()
        self.tab_widget.addTab(self.video_tab, "视频生成")
        
        layout.addWidget(self.tab_widget)
    
    def set_tab_index(self, index):
        """设置当前标签页索引"""
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index) 