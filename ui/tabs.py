import os
import json
import time
import traceback
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QTextEdit, QLabel, QProgressBar, QTableWidget, 
                           QTableWidgetItem, QMessageBox, QDialog, QSpinBox,
                           QDoubleSpinBox, QComboBox, QTabWidget, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from models.chatbot import ChatbotManager

# 检查matplotlib是否可用
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class ChatTab(QWidget):
    """聊天标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chatbot = ChatbotManager()
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        
        # 聊天历史区域
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)
        
        # 输入区域
        input_layout = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setMaximumHeight(100)
        input_layout.addWidget(self.input_box)
        
        # 发送按钮
        send_button = QPushButton("发送")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
        
    def send_message(self):
        """发送消息"""
        message = self.input_box.toPlainText().strip()
        if not message:
            return
            
        # 清空输入框
        self.input_box.clear()
        
        # 显示用户消息
        self.chat_history.append(f"你: {message}")
        
        # 获取机器人回复
        try:
            response = self.chatbot.get_response(message)
            self.chat_history.append(f"AI: {response}")
        except Exception as e:
            self.chat_history.append(f"AI: 抱歉，我现在无法回答。错误：{str(e)}")
        
        # 滚动到底部
        self.chat_history.moveCursor(QTextCursor.End)

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
        
    def start_training(self):
        """开始训练"""
        if not self.training_data:
            QMessageBox.warning(self, "警告", "没有训练数据，请先添加意图")
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
            QMessageBox.information(self, "训练完成", "模型训练已成功完成！")
        else:
            self.progress_bar.setValue(0)
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