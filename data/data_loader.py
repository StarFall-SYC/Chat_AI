#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载器模块 - 负责加载、转换和预处理训练数据
"""

import os
import json
import random
import traceback

def load_training_data(data_path=None, merge_extended=False):
    """
    加载训练数据
    
    Args:
        data_path (str): 训练数据文件路径，如果不提供则使用默认路径
        merge_extended (bool): 是否合并扩展训练数据
        
    Returns:
        list: 训练数据列表
    """
    try:
        # 获取默认路径
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'training', 'intents.json')
        
        # 加载主训练数据
        if not os.path.exists(data_path):
            print(f"训练数据文件不存在: {data_path}")
            return []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式，处理不同格式的JSON数据
        if isinstance(data, dict) and 'intents' in data:
            # 处理带有'intents'键的格式
            intents = data['intents']
        elif isinstance(data, list):
            # 直接列表格式
            intents = data
        else:
            print("训练数据格式不正确")
            return []
        
        # 合并扩展训练数据
        if merge_extended:
            # 获取扩展数据路径
            extended_path = os.path.join(os.path.dirname(data_path), 'extended_intents.json')
            if os.path.exists(extended_path):
                with open(extended_path, 'r', encoding='utf-8') as f:
                    extended_data = json.load(f)
                
                # 处理扩展数据的格式
                if isinstance(extended_data, dict) and 'intents' in extended_data:
                    extended_intents = extended_data['intents']
                elif isinstance(extended_data, list):
                    extended_intents = extended_data
                else:
                    extended_intents = []
                
                # 合并数据，避免重复
                existing_tags = {intent['tag'] for intent in intents}
                
                for ext_intent in extended_intents:
                    tag = ext_intent.get('tag')
                    if tag in existing_tags:
                        # 找到对应的意图
                        for intent in intents:
                            if intent['tag'] == tag:
                                # 合并模式，避免重复
                                existing_patterns = set(intent.get('patterns', []))
                                for pattern in ext_intent.get('patterns', []):
                                    if pattern not in existing_patterns:
                                        intent.setdefault('patterns', []).append(pattern)
                                
                                # 合并回复，避免重复
                                existing_responses = set(intent.get('responses', []))
                                for response in ext_intent.get('responses', []):
                                    if response not in existing_responses:
                                        intent.setdefault('responses', []).append(response)
                                break
                    else:
                        # 新意图，直接添加
                        intents.append(ext_intent)
                        existing_tags.add(tag)
                
                print(f"已合并扩展训练数据，共 {len(intents)} 个意图")
        
        return intents
    
    except Exception as e:
        print(f"加载训练数据时出错: {str(e)}")
        traceback.print_exc()
        return []

def perform_data_augmentation(training_data, augmentation_factor=1):
    """
    对训练数据进行简单的增强
    
    Args:
        training_data (list): 训练数据列表
        augmentation_factor (int): 每个样本的增强次数
    
    Returns:
        list: 增强后的训练数据
    """
    if augmentation_factor <= 0:
        return training_data
    
    # 创建副本，避免修改原始数据
    augmented_data = []
    for intent in training_data:
        augmented_intent = intent.copy()
        original_patterns = intent.get('patterns', [])
        
        # 创建新的模式列表，包含原始模式
        new_patterns = list(original_patterns)
        
        # 对每个原始模式进行增强
        for pattern in original_patterns:
            # 为每个模式生成augmentation_factor个变体
            for _ in range(augmentation_factor):
                augmented_pattern = simple_augment_text(pattern)
                if augmented_pattern and augmented_pattern not in new_patterns:
                    new_patterns.append(augmented_pattern)
        
        # 更新增强后的意图
        augmented_intent['patterns'] = new_patterns
        augmented_data.append(augmented_intent)
    
    return augmented_data

def simple_augment_text(text):
    """
    对文本进行简单的增强
    
    Args:
        text (str): 原始文本
    
    Returns:
        str: 增强后的文本
    """
    if not text:
        return ""
    
    # 选择一种增强方法
    augmentation_type = random.randint(1, 4)
    
    if augmentation_type == 1:
        # 随机调整标点符号
        if random.random() < 0.5:
            # 添加标点
            if text[-1] not in "?!.,，。？！":
                punctuations = ["？", "！", "。", "~", "…"]
                return text + random.choice(punctuations)
        else:
            # 删除尾部标点
            if text[-1] in "?!.,，。？！":
                return text[:-1]
    
    elif augmentation_type == 2:
        # 替换同义词（简单模拟）
        synonym_pairs = [
            ("你好", "您好"), ("早上好", "早安"), ("晚上好", "晚安"),
            ("谢谢", "感谢"), ("再见", "拜拜"), ("是的", "对的"),
            ("不", "否"), ("可以", "能够"), ("想要", "希望"),
            ("认为", "觉得"), ("看", "瞧"), ("听", "听见"),
            ("说", "讲"), ("吃", "食用"), ("喝", "饮用")
        ]
        
        for original, replacement in synonym_pairs:
            if original in text:
                return text.replace(original, replacement, 1)
    
    elif augmentation_type == 3:
        # 简单的词序调整（仅对短词组有效）
        words = text.split()
        if len(words) >= 4:
            # 选择两个相邻的词交换位置
            i = random.randint(0, len(words) - 2)
            words[i], words[i+1] = words[i+1], words[i]
            return " ".join(words)
    
    elif augmentation_type == 4:
        # 添加或删除填充词
        filler_words = ["嗯", "呃", "那个", "就是", "其实", "我想", "可能", "大概", "基本上", "差不多"]
        
        if random.random() < 0.5 and len(text) > 5:
            # 在句子开头添加填充词
            return random.choice(filler_words) + "," + text
        else:
            # 在句子中间添加填充词
            words = list(text)
            if len(words) >= 5:
                insert_pos = random.randint(2, len(words) - 2)
                words.insert(insert_pos, random.choice(filler_words))
                return "".join(words)
    
    # 如果没有成功应用任何增强，返回原始文本
    return text

def export_intents_statistics(training_data, output_file=None):
    """
    导出训练数据统计信息
    
    Args:
        training_data (list): 训练数据列表
        output_file (str, optional): 输出文件路径
    
    Returns:
        list: 统计信息列表
    """
    stats = []
    
    for intent in training_data:
        tag = intent.get('tag', '')
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', [])
        
        stats.append({
            'tag': tag,
            'patterns_count': len(patterns),
            'responses_count': len(responses),
            'patterns': patterns,
            'responses': responses
        })
    
    # 排序，按标签字母顺序
    stats.sort(key=lambda x: x['tag'])
    
    # 如果提供了输出文件路径，保存到文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"导出统计信息时出错: {str(e)}")
    
    return stats

def save_training_data(training_data, output_file=None):
    """
    保存训练数据到文件
    
    Args:
        training_data (list): 训练数据列表
        output_file (str, optional): 输出文件路径
    
    Returns:
        bool: 是否保存成功
    """
    try:
        # 获取默认路径
        if output_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_file = os.path.join(base_dir, 'data', 'training', 'intents.json')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return True
    
    except Exception as e:
        print(f"保存训练数据时出错: {str(e)}")
        traceback.print_exc()
        return False 