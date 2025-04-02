# AI聊天大模型

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow)

一个基于机器学习的智能对话系统，支持中文自然语言处理和多意图识别。

## 项目特点

- 📝 **多意图识别**：支持74种不同类型的对话意图，全面覆盖通用对话和专业领域
- 🖥️ **友好界面**：提供美观的图形用户界面，支持聊天和训练功能
- 🔄 **可扩展性**：允许添加新意图和训练数据
- 🚀 **快速响应**：平均响应时间小于0.01秒
- 📊 **数据分析**：支持训练数据统计和模型评估
- 🛠️ **多种工具**：提供命令行、GUI、快速训练等多种工具
- 🧠 **专业领域**：内置AI技术、上下文识别、情感对话等专业领域数据集
- 📈 **数据增强**：智能扩充训练样本，提高模型泛化能力

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

# 训练专业领域综合模型
python scripts/train_specialized_model.py --augment --factor 2
```

## 核心功能

- **智能对话**：基于随机森林分类的意图识别，支持多种对话场景
- **训练管理**：可视化训练数据编辑，参数可调的模型训练
- **数据增强**：自动扩充训练样本，提高模型性能
- **UI界面**：基于PyQt5的美观界面，支持消息气泡和历史记录
- **专业知识**：内置AI技术、深度学习、NLP等多个技术领域的问答能力
- **上下文理解**：支持多轮对话和上下文相关的交互模式
- **情感交互**：能够识别和回应用户的情感表达
- **训练数据扩展**：提供专业的数据合并和增强工具，支持同义词替换等技术

## 屏幕截图

*(此处可添加应用截图)*

## 项目结构

```
ai_chat_model/
├── data/                  # 数据相关
│   └── training/          # 训练数据集
├── models/                # 模型相关
├── ui/                    # 用户界面
├── scripts/               # 实用脚本
├── docs/                  # 项目文档
├── main.py                # 主程序入口
└── requirements.txt       # 项目依赖
```

完整的项目结构说明请参见 [项目结构文档](docs/PROJECT_STRUCTURE.md)。

## 训练数据集

项目包含多个专业领域的训练数据集：

- **基础交互**：问候、告别、感谢等基础对话
- **AI与技术**：人工智能、机器学习、深度学习等技术领域问答
- **上下文对话**：支持多轮交互的上下文相关对话模式
- **情感互动**：情感识别和适当回应的对话场景
- **问题解决**：实际问题的解决方案和建议

详细说明请参见 [训练数据指南](docs/TRAINING_DATA.md)。

## 专业领域模型

项目提供了专业领域综合模型，集成了多个领域知识：

```bash
# 训练专业领域综合模型
python scripts/train_specialized_model.py --augment

# 使用专业领域模型进行对话
python scripts/test_chatbot.py -m models/chatbot_specialized_model
```

## 数据扩展工具

项目提供了强大的数据扩展工具：

```bash
# 合并所有训练数据并应用数据增强
python scripts/extend_training_data.py --augment

# 合并特定领域数据
python scripts/extend_training_data.py --files ai_tech_intents.json contextual_intents.json

# 数据扩展和模型训练一体化流程
python scripts/extend_and_train.py --augment --factor 5 --test
```

## 详细文档

- [功能说明](docs/FEATURES.md) - 详细功能介绍和使用方法
- [快速入门](docs/QUICKSTART.md) - 快速上手指南和常用命令
- [项目总结](docs/SUMMARY.md) - 项目成就和未来规划
- [技术文档](docs/TECHNICAL.md) - 完整技术文档和架构说明
- [训练数据指南](docs/TRAINING_DATA.md) - 训练数据格式和扩展方法
- [项目优化总结](docs/PROJECT_OPTIMIZATION.md) - 项目结构优化和训练数据扩展记录
- [数据扩展指南](docs/DATA_EXPANSION.md) - 专业领域数据扩展和综合模型开发指南

## 常见问题

#### Q: 如何添加新的意图？
A: 可以编辑`data/training/extended_intents.json`文件，添加新的意图定义，或使用训练数据扩展工具。

#### Q: 如何提高模型准确率？
A: 可以增加训练数据量，使用数据增强工具，或尝试使用`--max-features 10000`等参数调整模型。

#### Q: 为什么有些意图识别率较低？
A: 这可能是由于训练数据不足或意图之间的相似性导致的，可以添加更多样化的训练数据。

#### Q: 如何使用专业领域模型？
A: 使用命令 `python scripts/test_chatbot.py -m models/chatbot_specialized_model` 加载专业领域模型，或在GUI界面中通过设置菜单加载。

## 如何贡献

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交变更 (`git commit -m '添加新特性'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 版本历史

- **1.0.0** (2023-04) - 首个正式版本发布
- **1.1.0** (2023-06) - 添加专业领域数据集和数据扩展工具
- **1.2.0** (2023-08) - 增加专业领域综合模型和上下文对话能力

## 许可证

本项目采用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者 - your-email@example.com

项目链接: [https://github.com/your-username/ai-chat-model](https://github.com/your-username/ai-chat-model)

## 功能特点

- 支持74种不同类型的对话意图，包括日常交流、专业领域、情感交互等
- 提供完整的训练数据集和专业领域数据集，覆盖AI技术、问题解决等领域
- 用户友好的图形界面，支持多种交互方式
- 简单易用的聊天界面，支持实时交互
- 完整的训练管理界面，支持数据管理和模型训练
- 提供多种训练数据格式支持
- 强大的数据集管理和模型训练功能

## 新增功能

### 增强的用户界面

- **调试模式**：在模型菜单中可以启用调试模式，查看意图识别过程和置信度
- **置信度阈值调整**：可以根据需要调整识别阈值，实现更精确的对话控制
- **数据集查看器**：可视化浏览和分析各个训练数据集，查看统计信息
- **数据增强工具**：一键进行数据增强，提升模型性能
- **专业领域模型训练**：通过UI直接训练专业领域模型

### 专业领域模型

现在支持使用专业领域模型，该模型能够处理更多专业领域的对话。

要训练专业领域模型，可以通过以下方式：

1. 在UI中选择"模型" > "训练专业领域模型"
2. 或者使用命令行：

```bash
python scripts/train_specialized_model.py --output complete_training_data.json --model-name chatbot_specialized_model
```

使用专业领域模型：

1. 在UI中选择"模型" > "加载模型" > "专业领域模型"
2. 或者使用命令行启动时指定模型：

```bash
python app.py -m models/chatbot_specialized_model.pth
```

## 使用方法

### 聊天界面

1. 启动应用程序后，默认进入聊天界面
2. 在底部输入框中输入问题或消息
3. 点击发送按钮或按回车键发送消息
4. AI会根据你的输入生成回复
5. 可以通过"文件"菜单保存聊天记录

### 调试功能

1. 在"模型"菜单中选择"模型设置" > "调试模式"启用调试
2. 调试模式下，聊天回复会包含意图识别和置信度信息
3. 在"模型"菜单中选择"模型设置" > "置信度阈值"调整识别阈值

### 数据管理

1. 在"数据"菜单中选择"数据集查看器"浏览训练数据
2. 可以查看各个数据集的统计信息和详细内容
3. 支持导出数据集到其他位置

### 数据增强

1. 在"数据"菜单中选择"数据增强"
2. 选择要增强的数据集文件
3. 设置增强因子（通常为2-5之间的值）
4. 指定输出文件名
5. 系统会自动进行数据增强并保存结果

### 训练管理

1. 切换到"训练"标签页
2. 加载训练数据（可以从"数据"菜单中选择）
3. 设置训练参数
4. 点击"开始训练"按钮启动训练
5. 训练完成后模型会自动保存

### 模型管理

1. 从"模型"菜单中选择"加载模型"可以切换不同的模型
2. "标准模型"适合一般对话
3. "专业领域模型"更适合专业话题讨论
4. 可以随时通过"训练专业领域模型"创建或更新专业模型 