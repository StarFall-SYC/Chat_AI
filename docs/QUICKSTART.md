# AI聊天大模型 - 快速入门指南

## 安装

1. 确保已安装Python 3.6+
2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```

## 启动命令一览

| 功能 | 命令 | 说明 |
|------|------|------|
| 完整应用 | `python scripts/run_demo.py` | 启动完整应用，包含聊天和训练功能 |
| 聊天界面 | `python scripts/start_chat.py` | 仅启动聊天界面 |
| 训练界面 | `python scripts/start_training.py` | 仅启动训练界面 |
| 命令行聊天 | `python scripts/test_chatbot.py` | 在命令行中测试聊天功能 |
| 调试聊天 | `python scripts/test_chatbot.py --debug` | 显示详细的意图预测信息 |
| 快速训练 | `python scripts/quick_train.py` | 使用扩展数据集训练模型 |
| 基础训练 | `python scripts/quick_train.py --basic` | 使用基础数据集训练模型 |
| 自定义训练 | `python scripts/quick_train.py --trees 200 --test-size 0.3` | 设置自定义训练参数 |

## 首次使用步骤

1. **训练模型**：
   ```
   python scripts/quick_train.py
   ```

2. **测试聊天功能**：
   ```
   python scripts/test_chatbot.py
   ```

3. **启动图形界面**：
   ```
   python scripts/start_chat.py
   ```

## 常用对话示例

| 说些什么... | AI可能的回复 |
|------------|-------------|
| 你好 | 嗨！很高兴见到你！ |
| 再见 | 再见，祝您有愉快的一天！ |
| 谢谢 | 不客气！ |
| 讲个笑话 | 为什么程序员不喜欢户外活动？因为户外有太多的bug。 |
| 天气怎么样 | 今天天气如何取决于你所在的位置，你可以看看窗外或使用天气应用。 |

## 文件结构速览

- `data/` - 数据相关文件
  - `training/` - 训练数据集
  - `models/` - 模型保存目录  
  - `history/` - 对话历史记录
- `models/` - 模型实现代码
- `ui/` - 用户界面实现
- `scripts/` - 功能脚本
  - `run_demo.py` - 完整应用启动脚本
  - `start_chat.py` - 聊天界面启动脚本
  - `start_training.py` - 训练界面启动脚本
  - `test_chatbot.py` - 命令行测试脚本
  - `test_trainer.py` - 训练测试脚本
  - `quick_train.py` - 快速训练脚本
- `docs/` - 项目文档

## 常见问题快速解决

- **模型未加载**：运行 `python scripts/quick_train.py` 训练模型
- **依赖包错误**：确保已运行 `pip install -r requirements.txt`
- **JSON格式错误**：检查 `data/training/*.json` 文件格式
- **GUI未显示**：确认已安装PyQt5 `pip install pyqt5`

## 查看更多

详细功能请参考 [功能说明](FEATURES.md) 和 [技术文档](TECHNICAL.md)。 