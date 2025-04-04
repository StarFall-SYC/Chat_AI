# 多模态训练数据加载指南

## 功能概述

AI聊天模型现在支持从文件夹批量加载多种类型的文件作为训练数据，包括JSON、文本文件、图像和视频。系统会自动识别不同类型的文件，并将它们转换为适合训练的格式，同时保持与多媒体内容的关联性，使模型能够在对话中返回相关的图像和视频内容。

## 支持的文件类型

- **JSON文件** (`.json`): 直接解析为训练数据结构
- **文本文件** (`.txt`, `.md`): 自动分析文本内容，创建相关意图和模式
- **图像文件** (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`): 自动生成与图像相关的查询模式
- **视频文件** (`.mp4`, `.avi`, `.mov`, `.mkv`): 自动生成与视频相关的查询模式

## 使用方法

### 1. 准备数据

将需要训练的文件组织到一个文件夹中。可以按照以下建议组织文件：

- 将所有JSON格式的意图数据文件放入文件夹
- 添加包含相关话题的文本文件
- 添加希望AI能够在对话中展示的图像
- 添加希望AI能够在对话中播放的视频

文件夹可以包含子文件夹，系统会递归搜索所有文件。建议按主题或功能将文件组织到不同的子文件夹中，使结构更清晰。

### 2. 使用训练界面加载数据

1. 打开应用程序，切换到"训练"标签页
2. 点击"从文件夹加载"按钮
3. 在弹出的对话框中选择包含训练数据的文件夹
4. 系统将开始扫描文件夹中的文件，这可能需要一些时间，取决于文件数量

### 3. 查看加载结果

加载完成后，界面将显示以下信息：

- 加载的意图总数
- 处理的文件数量（按类型分类）
- 成功处理的文件数量
- 失败的文件数量

在数据预览列表中，你可以查看每个加载的意图及其包含的模式数量。带有媒体标记（如[image]或[video]）的条目表示该意图关联了多媒体内容。

### 4. 训练模型

数据加载完成后，你可以设置训练参数并点击"开始训练"按钮。训练过程与常规训练相同，但模型将保存媒体资源的引用，使其能够在对话中返回相关的多媒体内容。

### 5. 保存训练结果

训练完成后，点击"保存模型"按钮将模型保存到指定位置。系统会同时保存：
- 训练好的模型文件 (`.pth`)
- 相关的训练数据 (`.json`)

## 技术细节

### 自动处理不同类型文件

#### JSON文件处理

系统支持两种格式的JSON文件：
- 直接的意图列表：`[{"tag": "标签1", "patterns": [...], "responses": [...]}, ...]`
- 带有"intents"键的对象：`{"intents": [{"tag": "标签1", ...}, ...]}`

#### 文本文件处理

对于文本文件，系统会：
1. 使用文件名（不含扩展名）作为意图标签
2. 按空行将文本分割成段落
3. 对超长段落（>200字符）进一步分割成句子
4. 过滤掉过长的句子（>500字符）
5. 使用最多30个处理后的段落/句子作为训练模式
6. 自动生成简单的回复文本

#### 图像文件处理

对于图像文件，系统会：
1. 使用文件名（不含扩展名）+"_image"作为标签
2. 自动生成与该图像相关的查询模式（如"显示xxx图片"）
3. 保存图像的完整路径，以便在对话中展示

#### 视频文件处理

对于视频文件，系统会：
1. 使用文件名（不含扩展名）+"_video"作为标签
2. 自动生成与该视频相关的查询模式（如"播放xxx视频"）
3. 保存视频的完整路径，以便在对话中播放

### 多线程加载

为防止在处理大量文件时界面卡顿，系统使用单独的线程进行数据加载。加载过程完成后，会通过信号机制更新界面。

## 使用建议

- **文件命名**: 使用有意义的文件名，因为它们将直接作为标签或标签的一部分
- **数据平衡**: 确保各类意图有相似数量的训练样本，避免模型偏向某一类
- **文件大小**: 避免使用过大的媒体文件，以防止内存问题
- **文件编码**: 确保文本文件使用UTF-8编码，避免中文显示问题
- **路径问题**: 避免在文件路径中使用特殊字符，以防止加载失败

## 故障排除

- **文件加载失败**: 检查文件编码、格式是否正确，以及是否有足够的访问权限
- **训练失败**: 可能是因为训练数据过大或格式不正确，尝试减少数据量或检查文件格式
- **媒体不显示**: 检查保存模型后媒体文件的路径是否发生变化，或文件是否已被移动/删除
- **加载速度慢**: 大量文件可能导致加载时间较长，考虑减少文件数量或按需加载特定类型的文件

---

通过这一功能，AI聊天模型能够更轻松地学习新知识并关联多媒体内容，大大提升了系统的实用性和互动性。 