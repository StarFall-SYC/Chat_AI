# AI聊天机器人文档

本目录包含AI聊天机器人项目的核心文档，为开发者和用户提供必要的指南和参考资料。

## 文档列表

- [快速入门指南](QUICKSTART.md) - 快速开始使用项目的步骤和基本操作
- [功能概览](FEATURES.md) - 详细的功能说明和技术实现
- [项目结构](PROJECT_STRUCTURE.md) - 项目目录结构和优化说明

## 训练数据说明

### 数据格式

AI聊天机器人使用JSON格式的训练数据，结构如下：

```json
[
  {
    "tag": "意图标签",
    "patterns": ["用户可能的输入1", "用户可能的输入2", ...],
    "responses": ["AI可能的回复1", "AI可能的回复2", ...]
  },
  ...
]
```

### 默认数据文件

默认的训练数据文件位于 `data/training/training_data.json`。

### 创建自定义数据

1. 按照上述格式创建或编辑训练数据JSON文件
2. 确保每个意图有充足的训练样本（建议至少10个patterns）
3. 对于每个意图提供多样化的回复选项
4. 保存文件后通过训练界面或训练脚本进行训练

### 扩展数据建议

为提高模型质量，建议：

- 为每个意图提供不同表达方式的样本
- 包括常见的拼写错误和语法变体
- 添加领域特定的术语和表达方式
- 确保样本之间有足够的差异性

## 模型训练参数

训练界面中可以调整以下参数：

- **训练轮数**: 控制训练迭代次数，通常在100-500之间
- **学习率**: 控制模型参数更新速度，推荐值为0.001
- **批大小**: 每次参数更新使用的样本数量，一般设置为16-64之间

## 命令行训练

除了使用GUI界面，也可以通过命令行训练模型：

```bash
python scripts/train.py
```

## 进一步开发建议

1. **增加训练数据**: 为现有意图添加更多样本，或添加新的意图
2. **调整模型参数**: 探索不同的训练参数组合
3. **实现深度学习模型**: 考虑使用LSTM或Transformer替代随机森林
4. **添加多语言支持**: 扩展模型以支持更多语言 