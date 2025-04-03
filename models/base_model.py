"""
基础模型模块
包含聊天模型的基础类定义和训练相关功能
"""

import os
import sys
import json
import random
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 导入日志
from .logger import app_logger

class PreprocessedText:
    """预处理文本类，用于存储和处理文本数据"""
    
    def __init__(self, 
                 raw_text: str = "", 
                 tokens: List[str] = None, 
                 vector: List[float] = None):
        """
        初始化预处理文本
        
        Args:
            raw_text: 原始文本
            tokens: 分词结果
            vector: 向量表示
        """
        self.raw_text = raw_text
        self.tokens = tokens or []
        self.vector = vector or []
        self.metadata = {}
        
    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
        
    def __str__(self) -> str:
        """字符串表示"""
        return f"PreprocessedText('{self.raw_text[:50]}...', tokens={len(self.tokens)}, vector={len(self.vector)})"


class TrainingData:
    """训练数据类，用于加载和处理训练数据"""
    
    def __init__(self, data_path: str = None):
        """
        初始化训练数据
        
        Args:
            data_path: 训练数据路径，应为JSON格式
        """
        self.data_path = data_path
        self.intents = []
        self.all_words = []
        self.tags = []
        self.xy_pairs = []  # (X, y) 训练数据对
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
            
    def load_data(self, data_path: str) -> bool:
        """
        加载训练数据
        
        Args:
            data_path: 训练数据路径，应为JSON格式
            
        Returns:
            是否成功加载
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持两种格式: {"intents": [...]} 或 直接 [...]
            if isinstance(data, dict) and "intents" in data:
                self.intents = data["intents"]
            elif isinstance(data, list):
                self.intents = data
            else:
                app_logger.error(f"不支持的训练数据格式: {data_path}")
                return False
                
            app_logger.info(f"成功加载训练数据: {data_path}, 共 {len(self.intents)} 个意图")
            return True
            
        except Exception as e:
            app_logger.error(f"加载训练数据失败: {data_path}, 错误: {str(e)}")
            return False
            
    def preprocess(self, text_processor) -> bool:
        """
        预处理训练数据
        
        Args:
            text_processor: 文本处理器，用于分词和向量化
            
        Returns:
            是否成功预处理
        """
        try:
            # 清空已有数据
            self.all_words = []
            self.tags = []
            self.xy_pairs = []
            
            # 处理所有模式
            for intent in self.intents:
                tag = intent.get('tag', '')
                if not tag:
                    continue
                    
                self.tags.append(tag)
                
                for pattern in intent.get('patterns', []):
                    # 分词
                    tokens = text_processor.tokenize(pattern)
                    self.all_words.extend(tokens)
                    
                    # 添加(X, y)对
                    self.xy_pairs.append((tokens, tag))
                    
            # 去重和排序
            self.all_words = sorted(list(set([word.lower() for word in self.all_words])))
            self.tags = sorted(list(set(self.tags)))
            
            app_logger.info(f"预处理完成: {len(self.all_words)} 个单词, {len(self.tags)} 个标签, {len(self.xy_pairs)} 个训练样本")
            return True
            
        except Exception as e:
            app_logger.error(f"预处理训练数据失败: {str(e)}")
            return False
    
    def generate_training_data(self, text_processor) -> Tuple[List[List[float]], List[int]]:
        """
        生成用于训练的数据
        
        Args:
            text_processor: 文本处理器，用于向量化
            
        Returns:
            X_train, y_train 训练数据
        """
        X_train = []
        y_train = []
        
        for (pattern_tokens, tag) in self.xy_pairs:
            # 将单词转换为向量
            bag = text_processor.vectorize(pattern_tokens, self.all_words)
            X_train.append(bag)
            
            # 标签对应的索引
            tag_idx = self.tags.index(tag)
            y_train.append(tag_idx)
            
        return X_train, y_train


class ChatModel:
    """聊天模型基类，定义了基本接口"""
    
    def __init__(self):
        """初始化聊天模型"""
        self.name = "BaseModel"
        self.version = "1.0.0"
        self.model_type = "base"
        self.model_path = None
        self.loaded = False
        
    def load(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否成功加载
        """
        self.model_path = model_path
        return True
        
    def save(self, model_path: str) -> bool:
        """
        保存模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否成功保存
        """
        self.model_path = model_path
        return True
        
    def predict(self, text: str, text_processor) -> Tuple[str, float]:
        """
        预测文本的意图
        
        Args:
            text: 输入文本
            text_processor: 文本处理器
            
        Returns:
            (预测标签, 置信度)
        """
        return "未知", 0.0


class ModelTrainer:
    """模型训练器，用于训练聊天模型"""
    
    def __init__(self, 
                 model_type: str = "neural", 
                 epochs: int = 100, 
                 learning_rate: float = 0.001, 
                 batch_size: int = 32):
        """
        初始化模型训练器
        
        Args:
            model_type: 模型类型，支持 "neural", "forest", "transformer"
            epochs: 训练轮次
            learning_rate: 学习率
            batch_size: 批次大小
        """
        self.model_type = model_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.training_data = None
        self.text_processor = None
        self.callbacks = []
        
    def train(self, 
              training_data: TrainingData, 
              text_processor, 
              callback = None) -> bool:
        """
        训练模型
        
        Args:
            training_data: 训练数据
            text_processor: 文本处理器
            callback: 回调函数，用于接收训练进度
            
        Returns:
            是否成功训练
        """
        self.training_data = training_data
        self.text_processor = text_processor
        
        if callback:
            self.callbacks.append(callback)
            
        if self.model_type == "neural":
            return self._train_neural_network()
        elif self.model_type == "forest":
            return self._train_random_forest()
        elif self.model_type == "transformer":
            return self._train_transformer()
        else:
            app_logger.error(f"不支持的模型类型: {self.model_type}")
            return False
            
    def _train_neural_network(self) -> bool:
        """训练神经网络模型"""
        app_logger.info("开始训练神经网络模型")
        
        # 检查PyTorch是否可用
        if not TORCH_AVAILABLE:
            app_logger.error("训练神经网络需要PyTorch，但未安装")
            return False
            
        try:
            # 生成训练数据
            X_train, y_train = self.training_data.generate_training_data(self.text_processor)
            
            # 转换为PyTorch张量
            X = torch.FloatTensor(X_train)
            y = torch.LongTensor(y_train)
            
            # 定义模型参数
            input_size = len(X_train[0])
            hidden_size = 8
            output_size = len(self.training_data.tags)
            
            # 创建模型
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # 训练模型
            for epoch in range(self.epochs):
                # 前向传播
                outputs = self.model(X)
                loss = criterion(outputs, y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    total = y.size(0)
                    correct = (predicted == y).sum().item()
                    accuracy = correct / total
                
                # 调用回调
                for callback in self.callbacks:
                    callback(epoch + 1, loss.item(), accuracy)
                    
                # 每20轮记录一次日志
                if (epoch + 1) % 20 == 0:
                    app_logger.info(f"轮次: {epoch + 1}/{self.epochs}, 损失: {loss.item():.4f}, 准确率: {accuracy:.4f}")
            
            app_logger.info("神经网络模型训练完成")
            return True
            
        except Exception as e:
            app_logger.error(f"神经网络模型训练失败: {str(e)}")
            return False
            
    def _train_random_forest(self) -> bool:
        """训练随机森林模型"""
        app_logger.info("开始训练随机森林模型")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 生成训练数据
            X_train, y_train = self.training_data.generate_training_data(self.text_processor)
            
            # 创建模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # 训练模型 (随机森林不需要迭代训练)
            self.model.fit(X_train, y_train)
            
            # 计算准确率
            accuracy = self.model.score(X_train, y_train)
            app_logger.info(f"随机森林模型训练完成，准确率: {accuracy:.4f}")
            
            # 调用回调
            for callback in self.callbacks:
                callback(1, 0.0, accuracy)  # 随机森林没有损失值
            
            return True
            
        except ImportError:
            app_logger.error("训练随机森林需要scikit-learn，但未安装")
            return False
        except Exception as e:
            app_logger.error(f"随机森林模型训练失败: {str(e)}")
            return False
            
    def _train_transformer(self) -> bool:
        """训练Transformer模型"""
        app_logger.info("开始训练Transformer模型")
        
        # 检查PyTorch是否可用
        if not TORCH_AVAILABLE:
            app_logger.error("训练Transformer需要PyTorch，但未安装")
            return False
            
        try:
            app_logger.info("Transformer模型训练暂未实现，将使用神经网络训练")
            return self._train_neural_network()
            
        except Exception as e:
            app_logger.error(f"Transformer模型训练失败: {str(e)}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        保存模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否成功保存
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型和词汇表
            model_data = {
                "model": self.model,
                "model_type": self.model_type,
                "all_words": self.training_data.all_words,
                "tags": self.training_data.tags
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            app_logger.info(f"模型保存成功: {model_path}")
            return True
            
        except Exception as e:
            app_logger.error(f"保存模型失败: {str(e)}")
            return False 