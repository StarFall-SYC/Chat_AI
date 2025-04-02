#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据扩展工具
功能：合并多个训练数据文件并应用数据增强技术
"""

import os
import sys
import json
import random
import argparse
from tqdm import tqdm
import jieba
import re

# 确保能够导入项目根目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training")

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def save_json_file(data, file_path):
    """保存JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"文件已保存至 {file_path}")
        return True
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {e}")
        return False

def merge_intents(files):
    """合并多个意图文件的内容"""
    merged_data = {"intents": []}
    intent_tags = set()
    
    for file_path in files:
        data = load_json_file(file_path)
        if not data or "intents" not in data:
            continue
            
        for intent in data["intents"]:
            # 如果意图标签已存在，合并patterns和responses
            if intent["tag"] in intent_tags:
                for existing_intent in merged_data["intents"]:
                    if existing_intent["tag"] == intent["tag"]:
                        # 合并patterns，去重
                        existing_patterns = set(existing_intent["patterns"])
                        existing_patterns.update(intent["patterns"])
                        existing_intent["patterns"] = list(existing_patterns)
                        
                        # 合并responses，去重
                        existing_responses = set(existing_intent["responses"])
                        existing_responses.update(intent["responses"])
                        existing_intent["responses"] = list(existing_responses)
                        break
            else:
                # 添加新意图
                intent_tags.add(intent["tag"])
                merged_data["intents"].append(intent)
    
    return merged_data

def apply_data_augmentation(intents_data, augmentation_factor=3, synonym_replace_prob=0.3):
    """应用数据增强技术生成更多训练样本"""
    augmented_data = {"intents": []}
    
    # 简单同义词替换词典（可以扩展）
    synonyms = {
        "什么": ["啥", "哪些", "什么样", "如何理解"],
        "如何": ["怎么", "怎样", "如何才能", "用什么方法"],
        "为什么": ["为何", "为啥", "什么原因", "出于什么考虑"],
        "能够": ["可以", "能否", "能不能", "是否可以"],
        "需要": ["必须", "需求", "必要", "要求"],
        "好": ["不错", "优秀", "很棒", "良好"],
        "使用": ["用", "采用", "运用", "应用"],
        "问题": ["疑问", "困惑", "难题", "挑战"],
        "帮助": ["协助", "帮忙", "辅助", "支持"],
        "工作": ["运行", "运作", "机制", "原理"],
        "原理": ["机制", "方式", "工作方式", "运行逻辑"],
        "技术": ["技能", "方法", "实现方式", "实现手段"],
        "区别": ["差异", "不同", "差别", "区分"],
        "优势": ["优点", "长处", "特点", "好处"],
        "应用": ["用途", "使用场景", "实际使用", "落地场景"],
        "例子": ["示例", "案例", "比如", "举例"],
        "有没有": ["是否有", "存不存在", "有没", "是否存在"],
        "怎么样": ["如何", "效果好吗", "好不好", "评价"]
    }
    
    print("正在生成增强数据...")
    for intent in tqdm(intents_data["intents"]):
        new_intent = {
            "tag": intent["tag"],
            "patterns": intent["patterns"].copy(),
            "responses": intent["responses"]
        }
        
        original_patterns = intent["patterns"].copy()
        
        # 对每个原始pattern生成多个变体
        for pattern in original_patterns:
            for _ in range(augmentation_factor):
                new_pattern = pattern
                words = list(jieba.cut(pattern))
                
                # 随机替换部分词为同义词
                for i, word in enumerate(words):
                    if word in synonyms and random.random() < synonym_replace_prob:
                        words[i] = random.choice(synonyms[word])
                
                new_pattern = "".join(words)
                
                # 确保不重复且不为空
                if new_pattern and new_pattern not in new_intent["patterns"]:
                    new_intent["patterns"].append(new_pattern)
                    
                # 随机词序变换（仅适用于特定问句）
                if "是什么" in pattern or "什么是" in pattern:
                    if "是什么" in pattern:
                        transformed = pattern.replace("是什么", "")
                        transformed = f"什么是{transformed}"
                        if transformed not in new_intent["patterns"]:
                            new_intent["patterns"].append(transformed)
                    elif "什么是" in pattern:
                        transformed = pattern.replace("什么是", "")
                        transformed = f"{transformed}是什么"
                        if transformed not in new_intent["patterns"]:
                            new_intent["patterns"].append(transformed)
        
        augmented_data["intents"].append(new_intent)
    
    return augmented_data

def analyze_data(intents_data):
    """分析数据集的统计信息"""
    intent_count = len(intents_data["intents"])
    pattern_count = sum(len(intent["patterns"]) for intent in intents_data["intents"])
    response_count = sum(len(intent["responses"]) for intent in intents_data["intents"])
    
    patterns_per_intent = pattern_count / intent_count if intent_count else 0
    responses_per_intent = response_count / intent_count if intent_count else 0
    
    print("\n数据集统计信息:")
    print(f"意图类别数: {intent_count}")
    print(f"总模式数: {pattern_count}")
    print(f"总响应数: {response_count}")
    print(f"每个意图的平均模式数: {patterns_per_intent:.2f}")
    print(f"每个意图的平均响应数: {responses_per_intent:.2f}")
    
    # 分析每个意图的样本数
    print("\n各意图样本数:")
    for intent in sorted(intents_data["intents"], key=lambda x: len(x["patterns"]), reverse=True):
        print(f"{intent['tag']}: {len(intent['patterns'])} 个样本, {len(intent['responses'])} 个响应")

def main():
    parser = argparse.ArgumentParser(description="训练数据合并与增强工具")
    parser.add_argument("--files", nargs="+", help="要合并的文件列表，如不指定则合并data/training目录下所有json文件")
    parser.add_argument("--output", default="merged_enhanced_intents.json", help="输出文件名")
    parser.add_argument("--augment", action="store_true", help="是否进行数据增强")
    parser.add_argument("--factor", type=int, default=3, help="数据增强因子，默认为3")
    args = parser.parse_args()
    
    # 如果未指定文件，获取目录下所有json文件
    if args.files:
        files = [f if os.path.isabs(f) else os.path.join(DATA_DIR, f) for f in args.files]
    else:
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    
    if not files:
        print("未找到任何JSON文件")
        return
    
    print(f"将合并以下文件: {[os.path.basename(f) for f in files]}")
    merged_data = merge_intents(files)
    
    # 打印原始数据的统计信息
    print("\n合并后的数据统计信息:")
    analyze_data(merged_data)
    
    # 应用数据增强
    if args.augment:
        print(f"\n正在应用数据增强，增强因子: {args.factor}...")
        augmented_data = apply_data_augmentation(merged_data, args.factor)
        print("\n数据增强后的统计信息:")
        analyze_data(augmented_data)
        output_data = augmented_data
    else:
        output_data = merged_data
    
    # 保存结果
    output_path = os.path.join(DATA_DIR, args.output)
    save_json_file(output_data, output_path)

if __name__ == "__main__":
    main() 