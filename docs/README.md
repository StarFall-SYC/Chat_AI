# AI聊天大模型

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow)

一个基于机器学习的智能对话系统，支持中文自然语言处理和多意图识别。

## 项目特点

- 📝 **多意图识别**：支持21种不同类型的对话意图
- 🖥️ **友好界面**：提供美观的图形用户界面，支持聊天和训练功能
- 🔄 **可扩展性**：允许添加新意图和训练数据
- 🚀 **快速响应**：平均响应时间小于0.01秒
- 📊 **数据分析**：支持训练数据统计和模型评估
- 🛠️ **多种工具**：提供命令行、GUI、快速训练等多种工具

## 安装使用

### 环境要求

- Python 3.6+
- 建议在虚拟环境中运行

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-username/ai-chat-model.git
cd ai-chat-model

# 安装依赖
pip install -r requirements.txt

# 训练模型
python scripts/quick_train.py
```

### 快速开始

```bash
# 启动完整应用（聊天+训练）
python scripts/run_demo.py

# 或仅启动聊天界面
python scripts/start_chat.py

# 或仅启动训练界面
python scripts/start_training.py

# 命令行聊天测试
python scripts/test_chatbot.py
```

## 核心功能

- **智能对话**：基于随机森林分类的意图识别，支持多种对话场景
- **训练管理**：可视化训练数据编辑，参数可调的模型训练
- **数据增强**：自动扩充训练样本，提高模型性能
- **UI界面**：基于PyQt5的美观界面，支持消息气泡和历史记录

## 屏幕截图

*(此处可添加应用截图)*

## 项目结构

```
ai_chat_model/
├── data/                  # 数据相关
├── models/                # 模型相关
├── ui/                    # 用户界面
├── scripts/               # 实用脚本
├── docs/                  # 项目文档
├── main.py                # 主程序入口
└── requirements.txt       # 项目依赖
```

完整的项目结构说明请参见 [项目结构文档](docs/PROJECT_STRUCTURE.md)。

## 详细文档

- [功能说明](docs/FEATURES.md) - 详细功能介绍和使用方法
- [快速入门](docs/QUICKSTART.md) - 快速上手指南和常用命令
- [项目总结](docs/SUMMARY.md) - 项目成就和未来规划
- [技术文档](docs/TECHNICAL.md) - 完整技术文档和架构说明

## 常见问题

#### Q: 如何添加新的意图？
A: 可以编辑`data/training/extended_intents.json`文件，添加新的意图定义。

#### Q: 如何提高模型准确率？
A: 可以增加训练数据量，或尝试使用`--trees 200`参数提高随机森林的树数量。

#### Q: 为什么有些意图识别率较低？
A: 这可能是由于训练数据不足或意图之间的相似性导致的，可以添加更多样化的训练数据。

## 如何贡献

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交变更 (`git commit -m '添加新特性'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 版本历史

- **1.0.0** (2023-04) - 首个正式版本发布

## 许可证

本项目采用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者 - your-email@example.com

项目链接: [https://github.com/your-username/ai-chat-model](https://github.com/your-username/ai-chat-model) 