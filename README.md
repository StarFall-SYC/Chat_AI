# AI聊天机器人

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow)

一个简单高效的AI聊天机器人项目，具有友好的图形用户界面和可训练的对话模型。

## 项目特点

- **简洁的代码结构**：经过优化的项目结构，文件精简，易于理解和维护
- **友好的用户界面**：基于PyQt5构建的现代化图形界面
- **可训练的对话模型**：支持自定义训练数据和模型参数
- **多种对话能力**：支持基础对话、专业领域问答和情感互动
- **易于扩展**：模块化设计，便于添加新功能

## 快速开始

### 安装

1. 克隆项目仓库
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

### 运行

```
python main.py
```

## 项目结构

```
ai_chat_model/
├── data/            # 数据目录
│   ├── models/      # 模型保存目录
│   └── training/    # 训练数据目录
├── docs/            # 文档目录
├── models/          # 模型相关代码
│   ├── chatbot.py   # 聊天机器人核心实现
│   └── text_processor.py # 文本处理器
├── scripts/         # 脚本目录
│   └── train.py     # 训练脚本
├── ui/              # UI相关代码
│   ├── main_window.py # 主窗口
│   ├── tabs.py      # 标签页实现
│   └── styles.qss   # 样式表
└── main.py          # 程序入口
```

## 主要功能

- **聊天界面**：与AI进行自然语言对话
- **训练界面**：管理训练数据，训练和优化模型
- **模型管理**：加载和切换不同的模型
- **数据扩展**：扩展和增强训练数据

## 技术实现

- **自然语言处理**：使用jieba分词和TF-IDF向量化
- **机器学习模型**：使用随机森林分类器进行意图分类
- **图形用户界面**：基于PyQt5构建
- **多线程处理**：训练过程在独立线程中运行

## 后续开发

- 深度学习模型集成
- 知识图谱支持
- 多语言支持
- 语音交互功能

## 许可证

MIT 许可证

## 联系方式

项目维护者 - your-email@example.com

项目链接: [https://github.com/your-username/ai-chat-model](https://github.com/your-username/ai-chat-model) 