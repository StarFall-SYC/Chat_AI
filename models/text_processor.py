#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理模块，提供分词、向量化等功能
"""

import re
import string
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    """文本处理器类，提供分词和文本向量化功能"""
    
    def __init__(self, max_features=5000):
        """初始化文本处理器"""
        # 初始化向量化器
        self.vectorizer = None
        self.vocabulary = []
        self.max_features = max_features
        
        # 初始化jieba分词
        jieba.setLogLevel(20)  # 设置jieba的日志级别，避免输出调试信息
    
    def tokenize(self, text):
        """
        对文本进行分词和预处理
        
        Args:
            text (str): 输入文本
            
        Returns:
            list: 分词结果列表
        """
        if not text:
            return []
            
        # 将文本转换为小写
        text = text.lower()
        
        # 移除标点符号和特殊字符
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # 使用jieba分词
        tokens = jieba.lcut(text)
        
        # 移除停用词和空白词
        tokens = [token for token in tokens if token.strip() and len(token) > 1]
        
        return tokens
    
    def fit_transform(self, texts):
        """
        训练向量化器并转换文本
        
        Args:
            texts (list): 文本列表
            
        Returns:
            numpy.ndarray: 向量化后的文本矩阵
        """
        # 处理文本
        processed_texts = []
        for text in texts:
            tokens = self.tokenize(text)
            processed_texts.append(' '.join(tokens))
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,  # 使用设置的特征数量
            min_df=2,  # 最小文档频率
            max_df=0.8,  # 最大文档频率
            sublinear_tf=True  # 使用次线性TF缩放
        )
        
        # 训练向量化器并转换文本
        X = self.vectorizer.fit_transform(processed_texts)
        
        # 保存词汇表
        self.vocabulary = list(self.vectorizer.vocabulary_.keys())
        
        return X.toarray()
    
    def transform(self, texts):
        """
        使用已训练的向量化器转换文本
        
        Args:
            texts (list or str): 文本或文本列表
            
        Returns:
            numpy.ndarray: 向量化后的文本矩阵
        """
        if not self.vectorizer:
            raise ValueError("向量化器尚未训练，请先调用fit_transform方法")
        
        # 处理单个文本的情况
        if isinstance(texts, str):
            texts = [texts]
        
        # 处理文本
        processed_texts = []
        for text in texts:
            tokens = self.tokenize(text)
            processed_texts.append(' '.join(tokens))
        
        # 转换文本
        X = self.vectorizer.transform(processed_texts)
        
        return X.toarray()
    
    def get_vocabulary(self):
        """
        获取词汇表
        
        Returns:
            list: 词汇表列表
        """
        return self.vocabulary
    
    def get_feature_names(self):
        """
        获取特征名称
        
        Returns:
            list: 特征名称列表
        """
        if not self.vectorizer:
            return []
        
        try:
            # scikit-learn 1.0+
            return self.vectorizer.get_feature_names_out()
        except AttributeError:
            # 低版本scikit-learn
            return self.vectorizer.get_feature_names()
    
    def calculate_similarity(self, text1, text2):
        """
        计算两段文本的余弦相似度
        
        Args:
            text1 (str): 第一段文本
            text2 (str): 第二段文本
            
        Returns:
            float: 相似度得分 (0-1)
        """
        if not self.vectorizer:
            raise ValueError("向量化器尚未训练，请先调用fit_transform方法")
        
        # 向量化文本
        vec1 = self.transform([text1])[0]
        vec2 = self.transform([text2])[0]
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2) 