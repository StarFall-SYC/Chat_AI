# AI聊天机器人

![版本](https://img.shields.io/badge/版本-2.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow)

一个功能强大的AI聊天机器人项目，配备现代化图形用户界面和多模态能力，支持文本、图像和视频生成。

## 当前为半成品

## 项目特点

- **多模态能力**：支持文本对话、图像生成和视频生成
- **多种数据加载方式**：支持从文件夹批量加载多种类型文件（JSON、文本、图像、视频）进行训练
- **记忆和上下文理解**：能够保持对话上下文，提供连贯的交流体验
- **情感分析**：自动分析用户输入的情感倾向，适当调整回复
- **友好的图形界面**：基于PyQt5构建的现代化图形界面，支持多媒体内容展示
- **可训练的对话模型**：支持自定义训练数据和模型参数
- **大型语言模型支持**：集成Transformer架构和预训练模型，如ChatGLM
- **易于扩展**：模块化设计，便于添加新功能

## 快速开始

### 安装

1. 克隆项目仓库
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
3. 设置环境：
   ```
   python scripts/setup_environment.py
   ```

### 运行

```
python main.py
```

## 项目结构

```
ai_chat_model/
├── data/                  # 数据目录
│   ├── models/            # 模型保存目录
│   ├── training/          # 训练数据目录
│   ├── history/           # 对话历史记录
│   └── generated/         # 生成的内容
│       ├── images/        # 生成的图像
│       └── videos/        # 生成的视频
├── docs/                  # 文档目录
├── models/                # 模型相关代码
│   ├── chatbot.py         # 聊天机器人核心实现
│   ├── text_processor.py  # 文本处理器
│   └── advanced_model.py  # 高级模型实现（多模态）
├── scripts/               # 脚本目录
│   ├── train.py           # 训练脚本
│   └── setup_environment.py # 环境设置脚本
├── ui/                    # UI相关代码
│   ├── main_window.py     # 主窗口
│   ├── tabs.py            # 基础标签页实现
│   ├── advanced_tabs.py   # 高级功能标签页
│   └── styles.qss         # 样式表
└── main.py                # 程序入口
```

## 主要功能

- **聊天界面**：与AI进行自然语言对话，支持上下文理解
- **训练界面**：
  - 管理训练数据，训练和优化模型
  - 支持从文件夹批量加载多种文件类型（JSON、文本、图像、视频）
  - 提供训练数据预览和统计信息
  - 保存训练好的模型和相关资源
- **模型管理**：加载和切换不同的模型，包括本地模型和大型预训练模型
- **多模态生成**：
  - **图像生成**：基于文本描述生成图像
  - **视频生成**：基于文本描述生成短视频
- **情感分析**：分析文本的情感倾向和主观性

## 技术实现

- **文本处理**：使用jieba分词和TF-IDF向量化
- **基础模型**：随机森林分类器进行意图分类
- **高级模型**：集成Transformer架构和预训练大型语言模型
- **多模态支持**：
  - 自动从不同类型文件中提取训练数据
  - 使用标签关联多媒体资源实现多模态响应
- **多模态生成**：
  - 图像：使用Stable Diffusion实现文本到图像的生成
  - 视频：基于图像序列生成动态视频
- **情感分析**：使用TextBlob进行情感极性和主观性分析
- **图形用户界面**：基于PyQt5构建，支持富文本、图像和视频展示
- **多线程处理**：生成和训练过程在独立线程中运行
- **线程安全机制**：使用锁机制确保多模态操作的线程安全

## 故障排除

### 模块导入错误

如果遇到以下错误：
```
ModuleNotFoundError: No module named 'ui.chat_tab'
```

这是因为UI模块结构发生了变化。解决方法是修改`ui/__init__.py`文件：

```python
"""
UI目录
包含AI聊天模型的用户界面组件
"""

# 导出主要组件
from .main_window import MainWindow
from .tabs import ChatTab, TrainingTab  # 从tabs.py而不是chat_tab.py导入
# 导出高级标签页组件
try:
    from .advanced_tabs import MediaGenerationTab, ImageGenerationTab, VideoGenerationTab
except ImportError:
    pass
```

### scikit-learn版本不匹配警告

如果看到类似以下警告：
```
InconsistentVersionWarning: Trying to unpickle estimator RandomForestClassifier from version 1.6.1 when using version 1.5.1
```

这是因为保存模型时使用的scikit-learn版本与当前运行环境的版本不同。一般情况下这不会影响基本功能，但如果遇到问题，建议:
1. 升级scikit-learn到对应版本：`pip install scikit-learn==1.6.1`
2. 或重新用当前环境训练模型

## 后续开发计划

- 更高质量的视频生成能力
- 语音识别和语音合成功能
- 多语言支持
- 移动端适配
- 云端部署支持

## 许可证

MIT 许可证

## 联系方式

项目维护者 - your-email@example.com

项目链接: [https://github.com/your-username/ai-chat-model](https://github.com/your-username/ai-chat-model) 