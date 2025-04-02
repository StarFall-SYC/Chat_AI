from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, 
                         QApplication, QSplitter, QAction, QMenuBar, 
                         QStatusBar, QMenu, QMessageBox, QDialog, 
                         QDialogButtonBox, QLabel, QVBoxLayout, QTextBrowser, 
                         QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

import sys
import os
import json
import platform

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot import ChatbotManager
from .tabs import ChatTab, TrainingTab

class AboutDialog(QDialog):
    """关于对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("关于AI聊天大模型")
        self.resize(400, 300)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 添加标题
        title = QLabel("AI聊天大模型")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 添加版本信息
        version = QLabel("版本 1.1.0")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        # 添加描述
        description = QTextBrowser()
        description.setOpenExternalLinks(True)
        description.setHtml("""
        <p align='center'>一个功能强大的AI聊天大模型应用程序，具有现代化的GUI界面。</p>
        <p align='center'>支持专业领域对话、上下文理解和情感交互。</p>
        <br>
        <p><b>功能特点：</b></p>
        <ul>
            <li>支持74种不同对话意图的聊天模型</li>
            <li>专业领域模型，覆盖AI技术、问题解决等专业话题</li>
            <li>用户友好的图形界面</li>
            <li>聊天界面支持实时交互</li>
            <li>训练界面支持数据管理和模型训练</li>
            <li>支持保存和加载训练数据</li>
            <li>支持数据增强和模型训练</li>
            <li>调试模式显示意图识别详情</li>
            <li>可调整的置信度阈值</li>
        </ul>
        <br>
        <p><b>最新更新：</b></p>
        <ul>
            <li>添加专业领域模型支持</li>
            <li>集成调试模式到界面</li>
            <li>添加数据增强功能</li>
            <li>支持自定义置信度阈值</li>
            <li>改进聊天体验和响应质量</li>
        </ul>
        <br>
        <p><b>系统信息：</b></p>
        <ul>
            <li>操作系统：%s</li>
            <li>Python版本：%s</li>
            <li>PyQt版本：%s</li>
        </ul>
        """ % (
            platform.platform(),
            platform.python_version(),
            "5.15" # 可以从PyQt模块中获取真实版本
        ))
        layout.addWidget(description)
        
        # 添加按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI聊天机器人")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 添加聊天标签页
        chat_tab = ChatTab()
        tab_widget.addTab(chat_tab, "聊天")
        
        # 添加训练标签页
        training_tab = TrainingTab()
        tab_widget.addTab(training_tab, "训练")
        
        layout.addWidget(tab_widget)
        
        # 创建聊天机器人管理器
        self.chatbot = ChatbotManager()
        
        # 调试模式标志
        self.debug_mode = False
        
        # 置信度阈值
        self.confidence_threshold = 0.3
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("准备就绪")
        
        # 尝试加载模型
        self.load_model()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu('文件')
        
        # 保存聊天记录
        save_action = QAction('保存聊天记录', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_chat)
        file_menu.addAction(save_action)
        
        # 退出
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 模型菜单
        model_menu = menu_bar.addMenu('模型')
        
        # 训练模型
        train_model_action = QAction("训练模型", self)
        train_model_action.triggered.connect(lambda: self.tabs.setCurrentWidget(self.training_tab) or self.training_tab.start_training())
        model_menu.addAction(train_model_action)
        
        # 加载模型
        load_action = QAction('加载模型', self)
        load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self.load_model)
        model_menu.addAction(load_action)
        
        # 加载特定模型子菜单
        load_specific_menu = model_menu.addMenu('加载特定模型')
        
        # 默认模型
        default_model_action = QAction('默认模型', self)
        default_model_action.triggered.connect(lambda: self.load_specific_model(None))
        load_specific_menu.addAction(default_model_action)
        
        # 专业领域模型
        specialized_model_action = QAction('专业领域模型', self)
        specialized_model_action.triggered.connect(
            lambda: self.load_specific_model(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                             'models', 'chatbot_specialized_model')))
        load_specific_menu.addAction(specialized_model_action)
        
        # 训练专业领域模型
        train_specialized_model_action = QAction("训练专业领域模型", self)
        train_specialized_model_action.triggered.connect(self.train_specialized_model)
        model_menu.addAction(train_specialized_model_action)
        
        # 模型信息
        model_info_action = QAction('模型信息', self)
        model_info_action.setShortcut('Ctrl+I')
        model_info_action.triggered.connect(self.show_model_info)
        model_menu.addAction(model_info_action)
        
        model_menu.addSeparator()
        
        # 模型设置子菜单
        settings_menu = model_menu.addMenu('模型设置')
        
        # 调试模式
        self.debug_mode_action = QAction('调试模式', self, checkable=True)
        self.debug_mode_action.setChecked(self.debug_mode)
        self.debug_mode_action.triggered.connect(self.toggle_debug_mode)
        settings_menu.addAction(self.debug_mode_action)
        
        # 置信度阈值子菜单
        threshold_menu = settings_menu.addMenu('置信度阈值')
        
        # 添加不同的阈值选项
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.threshold_actions = {}
        
        for threshold in thresholds:
            action = QAction(f'{threshold:.1f}', self, checkable=True)
            action.setChecked(abs(self.confidence_threshold - threshold) < 0.01)
            action.triggered.connect(lambda checked, t=threshold: self.set_confidence_threshold(t))
            threshold_menu.addAction(action)
            self.threshold_actions[threshold] = action
        
        # 数据菜单
        data_menu = menu_bar.addMenu('数据')
        
        # 加载训练数据
        load_data_action = QAction("加载训练数据", self)
        load_data_action.triggered.connect(lambda: self.training_tab.load_data())
        data_menu.addAction(load_data_action)
        
        # 保存训练数据
        save_data_action = QAction("保存训练数据", self)
        save_data_action.triggered.connect(lambda: self.training_tab.save_data())
        data_menu.addAction(save_data_action)
        
        # 数据集查看器
        dataset_viewer_action = QAction('数据集查看器', self)
        dataset_viewer_action.triggered.connect(self.show_dataset_viewer)
        data_menu.addAction(dataset_viewer_action)
        
        # 数据增强
        augment_action = QAction('数据增强', self)
        augment_action.triggered.connect(self.run_data_augmentation)
        data_menu.addAction(augment_action)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu('帮助')
        
        # 关于
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_model(self):
        """加载模型"""
        try:
            success = self.chatbot.load_model()
            if success:
                self.statusBar().showMessage("模型加载成功")
            else:
                self.statusBar().showMessage("模型未加载，请先训练模型")
        except Exception as e:
            self.statusBar().showMessage(f"加载模型出错: {str(e)}")
    
    def save_chat(self):
        """保存聊天记录"""
        if self.tabs.currentWidget() == self.chat_tab:
            self.chat_tab.export_chat()
        else:
            self.tabs.setCurrentWidget(self.chat_tab)
            self.chat_tab.export_chat()
    
    def show_about(self):
        """显示关于对话框"""
        dialog = AboutDialog(self)
        dialog.exec_()
    
    def load_specific_model(self, model_path):
        """加载指定的模型"""
        try:
            self.chatbot.custom_model_path = model_path
            success = self.chatbot.load_model()
            if success:
                self.statusBar().showMessage(f"模型 {model_path} 加载成功")
            else:
                self.statusBar().showMessage(f"模型 {model_path} 加载失败")
        except Exception as e:
            self.statusBar().showMessage(f"加载模型出错: {str(e)}")
    
    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        self.debug_mode_action.setChecked(self.debug_mode)
        
        # 调用ChatbotManager的方法设置调试模式
        if hasattr(self.chatbot, 'set_debug_mode'):
            self.chatbot.set_debug_mode(self.debug_mode)
            
        self.statusBar().showMessage(f"调试模式：{'已启用' if self.debug_mode else '已禁用'}")
    
    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = threshold
        
        # 调用ChatbotManager的方法设置置信度阈值
        if hasattr(self.chatbot, 'set_confidence_threshold'):
            success = self.chatbot.set_confidence_threshold(threshold)
            if success:
                self.statusBar().showMessage(f"置信度阈值已设置为: {threshold:.1f}")
            else:
                self.statusBar().showMessage(f"设置置信度阈值失败，请确保值在0-1之间")
    
    def run_data_augmentation(self):
        """运行数据增强"""
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        import subprocess
        import os
        
        # 获取输入文件
        file_options = ["contextual_intents.json", "ai_tech_intents.json", 
                        "extended_intents.json", "advanced_intents.json",
                        "problem_solving_intents.json", "emotional_intents.json",
                        "domain_specific_intents.json"]
        input_file, ok = QInputDialog.getItem(
            self, "选择输入文件", "请选择要增强的训练数据文件:", 
            file_options, 0, False
        )
        if not ok or not input_file:
            return
            
        # 获取增强因子
        factor, ok = QInputDialog.getInt(
            self, "设置增强因子", "请设置数据增强因子 (建议 2-5):", 
            2, 1, 10, 1
        )
        if not ok:
            return
            
        # 获取输出文件
        output_file, ok = QInputDialog.getText(
            self, "输出文件", "请输入增强后数据的保存文件名:", 
            text=f"augmented_{input_file}"
        )
        if not ok or not output_file:
            return
            
        try:
            # 构建命令
            cmd = [
                "python", 
                "scripts/extend_and_train.py",
                "--files", input_file,
                "--augment",
                "--factor", str(factor),
                "--output", output_file,
                "--max-features", "3000",
                "--skip-training"
            ]
            
            # 显示正在处理的消息
            self.statusBar().showMessage(f"正在处理数据增强...")
            
            # 执行命令
            process = subprocess.Popen(
                cmd, 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                QMessageBox.information(
                    self, 
                    "数据增强完成", 
                    f"数据增强成功完成！\n输出文件: {output_file}"
                )
                self.statusBar().showMessage(f"数据增强完成，输出至 {output_file}")
            else:
                QMessageBox.warning(
                    self, 
                    "数据增强失败", 
                    f"数据增强过程中发生错误:\n{stderr}"
                )
                self.statusBar().showMessage("数据增强失败")
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"执行数据增强时发生错误:\n{str(e)}"
            )
            self.statusBar().showMessage("数据增强发生错误")
    
    def train_specialized_model(self):
        """训练专业领域模型"""
        from PyQt5.QtWidgets import QMessageBox
        import subprocess
        import os
        
        try:
            # 构建命令
            cmd = [
                "python", 
                "scripts/train_specialized_model.py",
                "--output", "complete_training_data.json",
                "--model-name", "chatbot_specialized_model",
                "--max-features", "5000",
                "--n-estimators", "200"
            ]
            
            # 询问是否启用数据增强
            reply = QMessageBox.question(
                self, 
                '数据增强', 
                '是否在训练前启用数据增强？',
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                cmd.append("--augment")
            
            # 显示正在处理的消息
            self.statusBar().showMessage(f"正在训练专业领域模型...")
            
            # 执行命令
            process = subprocess.Popen(
                cmd, 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                QMessageBox.information(
                    self, 
                    "模型训练完成", 
                    f"专业领域模型训练成功完成！\n模型已保存至 models/chatbot_specialized_model.pth"
                )
                self.statusBar().showMessage(f"专业领域模型训练完成")
            else:
                QMessageBox.warning(
                    self, 
                    "模型训练失败", 
                    f"训练过程中发生错误:\n{stderr}"
                )
                self.statusBar().showMessage("模型训练失败")
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"执行模型训练时发生错误:\n{str(e)}"
            )
            self.statusBar().showMessage("模型训练发生错误")
    
    def show_dataset_viewer(self):
        """显示数据集查看器"""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                                 QListWidget, QTextEdit, QPushButton, 
                                 QLabel, QSplitter, QMessageBox)
        from PyQt5.QtCore import Qt
        import os
        import json
        
        class DatasetViewerDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("训练数据集查看器")
                self.resize(800, 600)
                
                # 获取训练数据目录
                self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.training_dir = os.path.join(self.base_path, 'data', 'training')
                
                # 创建主布局
                main_layout = QVBoxLayout(self)
                
                # 创建标题
                title = QLabel("训练数据集查看器")
                title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
                title.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(title)
                
                # 创建说明
                desc = QLabel("选择左侧的数据文件查看详情。每个数据集文件包含多个对话意图及其相关的训练样本。")
                desc.setWordWrap(True)
                main_layout.addWidget(desc)
                
                # 创建分割器
                splitter = QSplitter(Qt.Horizontal)
                main_layout.addWidget(splitter, 1)  # 1是拉伸因子
                
                # 创建左侧文件列表
                self.file_list = QListWidget()
                self.file_list.currentItemChanged.connect(self.load_selected_file)
                splitter.addWidget(self.file_list)
                
                # 创建右侧内容查看区
                self.content_view = QTextEdit()
                self.content_view.setReadOnly(True)
                splitter.addWidget(self.content_view)
                
                # 设置分割器比例
                splitter.setSizes([200, 600])
                
                # 创建底部按钮布局
                button_layout = QHBoxLayout()
                main_layout.addLayout(button_layout)
                
                # 添加刷新按钮
                refresh_button = QPushButton("刷新")
                refresh_button.clicked.connect(self.refresh_file_list)
                button_layout.addWidget(refresh_button)
                
                # 添加统计按钮
                stats_button = QPushButton("数据集统计")
                stats_button.clicked.connect(self.show_dataset_stats)
                button_layout.addWidget(stats_button)
                
                # 添加导出按钮
                export_button = QPushButton("导出数据")
                export_button.clicked.connect(self.export_dataset)
                button_layout.addWidget(export_button)
                
                # 添加关闭按钮
                close_button = QPushButton("关闭")
                close_button.clicked.connect(self.accept)
                button_layout.addWidget(close_button)
                
                # 加载文件列表
                self.refresh_file_list()
            
            def refresh_file_list(self):
                """刷新文件列表"""
                self.file_list.clear()
                self.content_view.clear()
                
                try:
                    # 获取所有JSON文件
                    files = [f for f in os.listdir(self.training_dir) if f.endswith('.json')]
                    files.sort()  # 按字母顺序排序
                    
                    # 添加到列表
                    for f in files:
                        self.file_list.addItem(f)
                    
                    # 如果有文件，选中第一个
                    if files:
                        self.file_list.setCurrentRow(0)
                except Exception as e:
                    self.content_view.setPlainText(f"加载文件列表出错: {str(e)}")
            
            def load_selected_file(self, current, previous):
                """加载选中的文件内容"""
                if not current:
                    return
                
                file_name = current.text()
                file_path = os.path.join(self.training_dir, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    # 格式化显示json
                    formatted_content = json.dumps(content, ensure_ascii=False, indent=2)
                    self.content_view.setPlainText(formatted_content)
                except Exception as e:
                    self.content_view.setPlainText(f"加载文件内容出错: {str(e)}")
            
            def show_dataset_stats(self):
                """显示当前数据集的统计信息"""
                if not self.file_list.currentItem():
                    QMessageBox.warning(self, "警告", "请先选择一个数据集文件")
                    return
                
                file_name = self.file_list.currentItem().text()
                file_path = os.path.join(self.training_dir, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 分析数据结构
                    intents = []
                    total_patterns = 0
                    total_responses = 0
                    
                    # 处理不同格式的训练数据
                    if isinstance(data, dict) and "intents" in data:
                        intents = data["intents"]
                    else:
                        intents = data
                    
                    # 统计每个意图的模式和响应数量
                    intent_stats = []
                    for intent in intents:
                        if isinstance(intent, dict) and 'tag' in intent:
                            patterns = intent.get('patterns', [])
                            responses = intent.get('responses', [])
                            
                            intent_stats.append({
                                'tag': intent['tag'],
                                'patterns': len(patterns),
                                'responses': len(responses)
                            })
                            
                            total_patterns += len(patterns)
                            total_responses += len(responses)
                    
                    # 构建统计信息
                    stats_text = f"数据集: {file_name}\n\n"
                    stats_text += f"总意图数: {len(intents)}\n"
                    stats_text += f"总模式数: {total_patterns}\n"
                    stats_text += f"总响应数: {total_responses}\n"
                    
                    if intent_stats:
                        avg_patterns = total_patterns / len(intent_stats)
                        avg_responses = total_responses / len(intent_stats)
                        stats_text += f"平均每个意图的模式数: {avg_patterns:.2f}\n"
                        stats_text += f"平均每个意图的响应数: {avg_responses:.2f}\n\n"
                    
                    stats_text += "各意图详细统计:\n"
                    for stat in intent_stats:
                        stats_text += f"- {stat['tag']}: {stat['patterns']} 模式, {stat['responses']} 响应\n"
                    
                    # 显示统计信息
                    QMessageBox.information(self, "数据集统计", stats_text)
                    
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"统计数据时发生错误: {str(e)}")
            
            def export_dataset(self):
                """导出当前数据集"""
                if not self.file_list.currentItem():
                    QMessageBox.warning(self, "警告", "请先选择一个数据集文件")
                    return
                
                from PyQt5.QtWidgets import QFileDialog
                
                file_name = self.file_list.currentItem().text()
                file_path = os.path.join(self.training_dir, file_name)
                
                # 打开保存对话框
                save_path, _ = QFileDialog.getSaveFileName(
                    self, 
                    "导出数据集", 
                    os.path.join(os.path.expanduser("~"), file_name),
                    "JSON文件 (*.json)"
                )
                
                if not save_path:
                    return
                
                try:
                    import shutil
                    shutil.copy2(file_path, save_path)
                    QMessageBox.information(self, "导出成功", f"数据集已导出至: {save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "导出失败", f"导出数据集时发生错误: {str(e)}")
        
        # 显示对话框
        dialog = DatasetViewerDialog(self)
        dialog.exec_()
    
    def show_model_info(self):
        """显示模型信息对话框"""
        # 尝试获取模型信息
        if not hasattr(self.chatbot, 'get_model_info'):
            QMessageBox.warning(self, "提示", "当前版本不支持获取模型信息")
            return
            
        model_info = self.chatbot.get_model_info()
        
        if not model_info:
            QMessageBox.warning(self, "提示", "模型尚未加载")
            return
            
        # 创建模型信息对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("模型信息")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # 创建表格来显示模型信息
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["属性", "值"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 填充表格
        row = 0
        for key, value in model_info.items():
            table.insertRow(row)
            
            # 美化键名
            display_key = key.replace("_", " ").title()
            
            # 设置单元格
            table.setItem(row, 0, QTableWidgetItem(display_key))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
            
            row += 1
            
        layout.addWidget(table)
        
        # 添加关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        # 显示对话框
        dialog.exec_() 