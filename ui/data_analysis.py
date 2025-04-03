"""
数据分析页面模块
提供训练数据可视化和分析功能
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QComboBox, QFileDialog, QTabWidget, QSplitter,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QCheckBox, QRadioButton, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from ui.custom_widgets import ModernButton, StyledCard, StatusInfoWidget

# 检查matplotlib是否可用
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import numpy as np
    
    # 配置matplotlib支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'STHeiti', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 确保models目录在sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入models中的模块
try:
    from models import app_logger, lock_manager
    from models.lock_manager import ResourceLock
except ImportError:
    # 创建占位符
    class DummyLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    app_logger = DummyLogger()
    
    # 创建占位符锁管理器
    class DummyLockManager:
        def acquire_lock(self, resource_id, category="model", timeout=-1): return True
        def release_lock(self, resource_id, category="model"): return True
    lock_manager = DummyLockManager()
    
    # 创建占位符资源锁
    class ResourceLock:
        def __init__(self, resource_id, category="model", timeout=-1):
            self.resource_id = resource_id
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

class DataAnalysisTab(QWidget):
    """数据分析标签页"""
    
    def __init__(self, parent=None):
        """初始化"""
        super().__init__(parent)
        self.data = []
        self.current_chart = "bar"
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 创建顶部标题区域
        header_widget = StyledCard(self)
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建顶部标题
        title_label = QLabel("训练数据分析")
        title_label.setProperty("title", "true")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        header_layout.addWidget(title_label)
        
        # 创建描述文本
        desc_text = (
            "本页面用于分析和可视化训练数据，帮助您了解数据的质量和分布情况。"
            "您可以加载训练数据文件，查看数据统计信息，以及使用不同图表可视化数据。"
        )
        desc_label = QLabel(desc_text)
        desc_label.setProperty("subtitle", "true")
        desc_label.setWordWrap(True)
        header_layout.addWidget(desc_label)
        
        header_widget.layout.addLayout(header_layout)
        main_layout.addWidget(header_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # 1是拉伸因子
        
        # 创建左侧面板
        left_panel = StyledCard(self)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        # 创建数据管理组
        data_group = QGroupBox("数据管理")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)
        
        # 添加数据加载按钮
        load_button = ModernButton("加载训练数据")
        load_button.setIcon(QIcon("ui/icons/app_icon_16.png"))  # 使用现有图标
        load_button.clicked.connect(self.load_data)
        data_layout.addWidget(load_button)
        
        # 添加数据源下拉框
        source_layout = QFormLayout()
        source_layout.setSpacing(8)
        
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("选择数据源")
        self.data_source_combo.addItem("训练数据.json")
        self.data_source_combo.addItem("contextual_intents.json")
        self.data_source_combo.addItem("ai_tech_intents.json")
        self.data_source_combo.addItem("问题解决领域.json")
        self.data_source_combo.addItem("专业领域.json")
        self.data_source_combo.currentIndexChanged.connect(self.on_data_source_changed)
        source_layout.addRow("数据源:", self.data_source_combo)
        
        data_layout.addLayout(source_layout)
        
        # 添加数据统计标签
        self.stats_label = StatusInfoWidget("未加载数据", StatusInfoWidget.INFO)
        data_layout.addWidget(self.stats_label)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        # 创建可视化选项组
        viz_group = QGroupBox("可视化选项")
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(10)
        
        # 添加图表类型选择
        chart_layout = QFormLayout()
        chart_layout.setSpacing(8)
        
        self.chart_combo = QComboBox()
        self.chart_combo.addItem("柱状图")
        self.chart_combo.addItem("饼图")
        self.chart_combo.addItem("折线图")
        self.chart_combo.addItem("散点图")
        self.chart_combo.addItem("热力图")
        self.chart_combo.currentIndexChanged.connect(self.on_chart_type_changed)
        chart_layout.addRow("图表类型:", self.chart_combo)
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("意图数量")
        self.metric_combo.addItem("模式数量")
        self.metric_combo.addItem("响应数量")
        self.metric_combo.addItem("字词频率")
        self.metric_combo.currentIndexChanged.connect(self.on_metric_changed)
        chart_layout.addRow("度量指标:", self.metric_combo)
        
        viz_layout.addLayout(chart_layout)
        
        # 添加高级选项
        options_layout = QHBoxLayout()
        options_layout.setSpacing(10)
        
        self.normalize_check = QCheckBox("标准化数据")
        self.normalize_check.stateChanged.connect(self.update_chart)
        options_layout.addWidget(self.normalize_check)
        
        self.colorful_check = QCheckBox("使用彩色主题")
        self.colorful_check.setChecked(True)
        self.colorful_check.stateChanged.connect(self.update_chart)
        options_layout.addWidget(self.colorful_check)
        
        viz_layout.addLayout(options_layout)
        
        # 添加更新按钮
        update_button = ModernButton("更新图表", primary=False)
        update_button.clicked.connect(self.update_chart)
        viz_layout.addWidget(update_button)
        
        viz_group.setLayout(viz_layout)
        left_layout.addWidget(viz_group)
        
        # 添加导出按钮
        export_button = ModernButton("导出分析报告")
        export_button.setIcon(QIcon("ui/icons/app_icon_16.png"))  # 使用现有图标
        export_button.clicked.connect(self.export_report)
        left_layout.addWidget(export_button)
        
        # 添加左侧面板到分割器
        splitter.addWidget(left_panel)
        
        # 创建右侧面板
        right_panel = QTabWidget()
        right_panel.setDocumentMode(True)
        
        # 创建可视化标签页
        self.viz_tab = StyledCard(self)
        # 检查matplotlib是否可用
        if MATPLOTLIB_AVAILABLE:
            # 创建图表画布
            self.figure = plt.figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.viz_tab.addWidget(self.canvas)
            
            # 创建一个初始的空图表
            self.create_empty_chart()
        else:
            # Matplotlib不可用时显示提示
            not_available = QLabel("matplotlib未安装，无法显示图表")
            not_available.setAlignment(Qt.AlignCenter)
            not_available.setProperty("class", "warning-message")
            self.viz_tab.addWidget(not_available)
            
        right_panel.addTab(self.viz_tab, "可视化")
        
        # 创建数据表格标签页
        self.table_tab = StyledCard(self)
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(["意图", "模式数量", "响应数量", "示例模式"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setAlternatingRowColors(True)
        self.table_tab.addWidget(self.data_table)
        
        right_panel.addTab(self.table_tab, "数据表格")
        
        # 创建统计标签页
        self.stats_tab = StyledCard(self)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        self.stats_tab.addWidget(self.stats_text)
        
        right_panel.addTab(self.stats_tab, "统计信息")
        
        # 添加右侧面板到分割器
        splitter.addWidget(right_panel)
        
        # 设置分割器的初始大小
        splitter.setSizes([300, 700])
    
    def load_data(self):
        """加载训练数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据文件", "", "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查数据结构
            if isinstance(data, dict) and "intents" in data:
                self.data = data["intents"]
            elif isinstance(data, list):
                self.data = data
            else:
                raise ValueError("不支持的JSON格式，应为意图列表或包含intents键的对象")
                
            # 更新UI
            self.update_stats()
            self.update_table()
            self.update_chart()
            
            # 更新数据源下拉框
            file_name = os.path.basename(file_path)
            if file_name not in [self.data_source_combo.itemText(i) for i in range(self.data_source_combo.count())]:
                self.data_source_combo.addItem(file_name)
            self.data_source_combo.setCurrentText(file_name)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
    
    def on_data_source_changed(self, index):
        """处理数据源变更"""
        if index == 0:  # "选择数据源"
            return
            
        # 获取选中的数据源
        source = self.data_source_combo.currentText()
        
        # 模拟加载不同数据源的数据
        if source != "选择数据源":
            # 在实际应用中，这里应该加载真实的数据
            # 这里仅作演示，生成模拟数据
            self.generate_mock_data()
            
            # 更新UI
            self.update_stats()
            self.update_table()
            self.update_chart()
    
    def generate_mock_data(self):
        """生成模拟数据（仅用于演示）"""
        # 清空现有数据
        self.data = []
        
        # 生成一些模拟意图
        intents = ["问候", "告别", "查询", "帮助", "投诉", "预订", "取消", "支付", "退款", "评价"]
        
        # 为每个意图生成模拟数据
        for intent in intents:
            # 随机生成模式和响应数量
            pattern_count = random.randint(3, 15)
            response_count = random.randint(2, 8)
            
            # 生成模式
            patterns = [f"{intent}模式{i+1}" for i in range(pattern_count)]
            
            # 生成响应
            responses = [f"{intent}响应{i+1}" for i in range(response_count)]
            
            # 添加到数据中
            self.data.append({
                "tag": intent,
                "patterns": patterns,
                "responses": responses
            })
    
    def update_stats(self):
        """更新统计信息"""
        if not self.data:
            self.stats_label.setText("未加载数据")
            self.stats_label.set_type(StatusInfoWidget.WARNING)
            return
        
        # 分析数据
        intent_count = len(self.data)
        pattern_count = sum(len(intent.get("patterns", [])) for intent in self.data)
        response_count = sum(len(intent.get("responses", [])) for intent in self.data)
        
        # 计算平均每个意图的模式和响应数量
        avg_patterns = pattern_count / intent_count if intent_count > 0 else 0
        avg_responses = response_count / intent_count if intent_count > 0 else 0
        
        # 更新简短统计显示
        stats_short = f"已加载 {intent_count} 个意图, {pattern_count} 个模式, {response_count} 个响应"
        self.stats_label.setText(stats_short)
        self.stats_label.set_type(StatusInfoWidget.SUCCESS)
        
        # 生成详细统计信息
        stats_text = f"""
<h3>数据统计分析</h3>

<b>基本信息：</b>
- 意图总数: {intent_count}
- 模式总数: {pattern_count}
- 响应总数: {response_count}
- 平均每个意图的模式数: {avg_patterns:.2f}
- 平均每个意图的响应数: {avg_responses:.2f}

<b>数据分布：</b>
"""
        
        # 计算模式长度分布
        pattern_lengths = []
        for intent in self.data:
            for pattern in intent.get("patterns", []):
                pattern_lengths.append(len(pattern))
        
        if pattern_lengths:
            avg_pattern_length = sum(pattern_lengths) / len(pattern_lengths)
            max_pattern_length = max(pattern_lengths) if pattern_lengths else 0
            min_pattern_length = min(pattern_lengths) if pattern_lengths else 0
            
            stats_text += f"""
- 平均模式长度: {avg_pattern_length:.2f} 字符
- 最长模式: {max_pattern_length} 字符
- 最短模式: {min_pattern_length} 字符
"""
        
        # 添加数据质量评估
        stats_text += """
<b>数据质量评估：</b>
"""
        
        # 评估模式数量是否足够
        if avg_patterns < 5:
            stats_text += "- 警告: 平均每个意图的模式数较少，建议增加更多训练样本\n"
        else:
            stats_text += "- 模式数量充足\n"
            
        # 评估响应数量是否足够
        if avg_responses < 3:
            stats_text += "- 警告: 平均每个意图的响应数较少，可能导致回复单一\n"
        else:
            stats_text += "- 响应数量充足\n"
            
        # 评估模式长度的合理性
        if pattern_lengths and avg_pattern_length < 10:
            stats_text += "- 警告: 模式长度较短，可能导致分类不准确\n"
        elif pattern_lengths and avg_pattern_length > 100:
            stats_text += "- 警告: 模式长度较长，可能包含无关信息\n"
        else:
            stats_text += "- 模式长度合理\n"
        
        # 更新统计文本区域
        self.stats_text.setHtml(stats_text)
    
    def update_table(self):
        """更新数据表格"""
        if not self.data:
            self.data_table.setRowCount(0)
            return
            
        # 设置表格行数
        self.data_table.setRowCount(len(self.data))
        
        # 填充表格
        for row, intent in enumerate(self.data):
            # 意图标签
            self.data_table.setItem(row, 0, QTableWidgetItem(intent.get("tag", "")))
            
            # 模式数量
            patterns = intent.get("patterns", [])
            self.data_table.setItem(row, 1, QTableWidgetItem(str(len(patterns))))
            
            # 响应数量
            responses = intent.get("responses", [])
            self.data_table.setItem(row, 2, QTableWidgetItem(str(len(responses))))
            
            # 示例模式
            if patterns:
                self.data_table.setItem(row, 3, QTableWidgetItem(patterns[0]))
    
    def on_chart_type_changed(self, index):
        """处理图表类型变更"""
        chart_types = ["bar", "pie", "line", "scatter", "heatmap"]
        if index < len(chart_types):
            self.current_chart = chart_types[index]
            self.update_chart()
    
    def on_metric_changed(self, index):
        """处理度量指标变更"""
        self.update_chart()
    
    def update_chart(self):
        """更新图表"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        if not self.data:
            self.create_empty_chart()
            return
            
        # 清除当前图表
        self.figure.clear()
        
        # 获取选中的度量指标
        metric_index = self.metric_combo.currentIndex()
        
        # 准备数据
        labels = [intent.get("tag", f"意图{i}") for i, intent in enumerate(self.data)]
        
        if metric_index == 0:  # 意图数量
            values = [1 for _ in self.data]
            title = "意图数量"
            ylabel = "数量"
        elif metric_index == 1:  # 模式数量
            values = [len(intent.get("patterns", [])) for intent in self.data]
            title = "每个意图的训练样本数量"
            ylabel = "训练样本数量"
        elif metric_index == 2:  # 响应数量
            values = [len(intent.get("responses", [])) for intent in self.data]
            title = "每个意图的响应数量"
            ylabel = "响应数量"
        elif metric_index == 3:  # 字词频率
            # 分析所有模式中的字词频率（简化实现）
            values = [
                sum([len(pattern) for pattern in intent.get("patterns", [])]) 
                for intent in self.data
            ]
            title = "每个意图的字词总数"
            ylabel = "字词数量"
        else:
            values = [1 for _ in self.data]
            title = "数据分析"
            ylabel = "数量"
        
        # 如果选择了标准化，对数据进行归一化处理
        if self.normalize_check.isChecked() and sum(values) > 0:
            values = [v / sum(values) for v in values]
            ylabel = "比例"
        
        # 选择颜色主题
        if self.colorful_check.isChecked():
            colors = plt.cm.tab10(np.arange(len(values)) / len(values))
        else:
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(values)))
        
        # 创建图表
        ax = self.figure.add_subplot(111)
        
        if self.current_chart == "bar":
            # 绘制柱状图
            bars = ax.bar(labels, values, color=colors)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}' if self.normalize_check.isChecked() else f'{int(height)}',
                        ha='center', va='bottom', rotation=0)
                
        elif self.current_chart == "pie":
            # 绘制饼图
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.axis('equal')  # 确保饼图是圆的
            
        elif self.current_chart == "line":
            # 绘制折线图
            ax.plot(labels, values, 'o-', color='b')
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}' if self.normalize_check.isChecked() else f'{int(v)}',
                        ha='center', va='bottom')
                
        elif self.current_chart == "scatter":
            # 绘制散点图
            ax.scatter(labels, values, s=100, c=colors, alpha=0.6)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}' if self.normalize_check.isChecked() else f'{int(v)}',
                        ha='center', va='bottom')
                
        elif self.current_chart == "heatmap":
            # 为热力图创建一个二维矩阵
            # 简化起见，使用一行数据
            matrix = np.array([values])
            im = ax.imshow(matrix, cmap='YlOrRd')
            
            # 添加颜色条
            self.figure.colorbar(im, ax=ax)
            
            # 设置标签
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks([0])
            ax.set_yticklabels(["值"])
            
            # 在每个单元格中添加数值
            for i in range(len(labels)):
                text = f'{values[i]:.2f}' if self.normalize_check.isChecked() else f'{int(values[i])}'
                ax.text(i, 0, text, ha="center", va="center", color="black")
        
        ax.set_title(title)
        self.figure.tight_layout()
        
        # 刷新画布
        self.canvas.draw()
    
    def create_empty_chart(self):
        """创建空图表"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "请加载数据以显示图表", ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
    
    def export_report(self):
        """导出分析报告"""
        if not self.data:
            QMessageBox.warning(self, "警告", "无数据可供导出")
            return
            
        # 打开保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出分析报告", "", "HTML文件 (*.html)"
        )
        
        if not file_path:
            return
            
        try:
            # 创建HTML报告内容
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>训练数据分析报告</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .stats-box { background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                    .warning { color: #e74c3c; }
                    .good { color: #2ecc71; }
                </style>
            </head>
            <body>
                <h1>训练数据分析报告</h1>
                <p>生成时间: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <h2>数据概况</h2>
                <div class="stats-box">
            """
            
            # 添加基本统计信息
            intent_count = len(self.data)
            pattern_count = sum(len(intent.get("patterns", [])) for intent in self.data)
            response_count = sum(len(intent.get("responses", [])) for intent in self.data)
            
            html += f"<p>意图总数: <strong>{intent_count}</strong></p>"
            html += f"<p>训练样本总数: <strong>{pattern_count}</strong></p>"
            html += f"<p>响应总数: <strong>{response_count}</strong></p>"
            
            # 计算平均值
            avg_patterns = pattern_count / intent_count if intent_count > 0 else 0
            avg_responses = response_count / intent_count if intent_count > 0 else 0
            
            html += f"<p>平均每个意图的训练样本数: <strong>{avg_patterns:.2f}</strong></p>"
            html += f"<p>平均每个意图的响应数: <strong>{avg_responses:.2f}</strong></p>"
            
            # 数据质量评估
            html += "<h2>数据质量评估</h2><div class='stats-box'>"
            
            if avg_patterns < 5:
                html += "<p class='warning'>⚠️ 训练样本数量偏少，建议增加训练样本</p>"
            else:
                html += "<p class='good'>✅ 训练样本数量充足</p>"
                
            if avg_responses < 3:
                html += "<p class='warning'>⚠️ 响应数量偏少，建议增加响应多样性</p>"
            else:
                html += "<p class='good'>✅ 响应数量充足</p>"
                
            html += "</div>"
            
            # 添加意图详情表格
            html += """
                <h2>意图详情</h2>
                <table>
                    <tr>
                        <th>意图</th>
                        <th>训练样本数量</th>
                        <th>响应数量</th>
                        <th>示例训练样本</th>
                    </tr>
            """
            
            for intent in self.data:
                patterns = intent.get("patterns", [])
                responses = intent.get("responses", [])
                
                html += f"<tr>"
                html += f"<td>{intent.get('tag', '')}</td>"
                html += f"<td>{len(patterns)}</td>"
                html += f"<td>{len(responses)}</td>"
                html += f"<td>{patterns[0] if patterns else ''}</td>"
                html += f"</tr>"
                
            html += "</table>"
            
            # 添加建议
            html += """
                <h2>改进建议</h2>
                <div class="stats-box">
            """
            
            if avg_patterns < 5:
                html += "<p>1. 为每个意图增加更多训练样本，特别是样本较少的意图。</p>"
                html += "<p>2. 尝试使用数据增强技术自动生成更多训练样本。</p>"
                
            if avg_responses < 3:
                html += "<p>3. 增加响应的多样性，避免机器人回复过于单调。</p>"
                
            if max(len(intent.get("patterns", [])) for intent in self.data) / (min(len(intent.get("patterns", [])) for intent in self.data) if min(len(intent.get("patterns", [])) for intent in self.data) > 0 else 1) > 5:
                html += "<p>4. 平衡各个意图的训练样本数量，避免数据不平衡问题。</p>"
                
            html += """
                </div>
            </body>
            </html>
            """
            
            # 保存HTML文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            QMessageBox.information(self, "导出成功", f"分析报告已导出至: {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出报告失败: {str(e)}") 