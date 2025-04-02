#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据扩展和模型训练一体化流程
功能：合并数据、应用数据增强、训练新模型并评估
"""

import os
import sys
import json
import time
import pickle
import subprocess
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 确保能够导入项目根目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入数据扩展工具
from scripts.extend_training_data import merge_intents, apply_data_augmentation, analyze_data, save_json_file
# 导入文本处理器
from models.text_processor import TextProcessor

# 数据和模型路径
DATA_DIR = os.path.join(root_dir, "data", "training")
MODEL_DIR = os.path.join(root_dir, "models")

def ensure_dir_exists(directory):
    """确保目录存在，如不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def get_data_files(specified_files=None, ext=".json"):
    """获取训练数据文件列表"""
    if specified_files:
        return [f if os.path.isabs(f) else os.path.join(DATA_DIR, f) for f in specified_files]
    else:
        return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(ext)]

def extend_data(args):
    """扩展训练数据"""
    print("\n===== 开始扩展训练数据 =====")
    
    # 获取文件列表
    data_files = get_data_files(args.files)
    if not data_files:
        print("未找到任何训练数据文件!")
        return None
    
    print(f"将处理以下文件: {[os.path.basename(f) for f in data_files]}")
    
    # 合并数据
    merged_data = merge_intents(data_files)
    print("\n合并后的数据统计:")
    analyze_data(merged_data)
    
    # 如果需要数据增强
    if args.augment:
        print(f"\n正在应用数据增强，增强因子: {args.factor}")
        augmented_data = apply_data_augmentation(merged_data, args.factor)
        print("\n数据增强后的统计:")
        analyze_data(augmented_data)
        final_data = augmented_data
    else:
        final_data = merged_data
    
    # 保存处理后的数据
    output_path = os.path.join(DATA_DIR, args.output)
    save_json_file(final_data, output_path)
    
    return output_path

def load_training_data(file_path):
    """加载训练数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("intents", [])
    except Exception as e:
        print(f"无法加载训练数据: {e}")
        return None

def train_model(training_file, args):
    """训练模型"""
    print("\n===== 开始训练模型 =====")
    
    # 加载训练数据
    intents_data = load_training_data(training_file)
    if not intents_data:
        print(f"无法加载训练数据: {training_file}")
        return None
    
    # 设置模型名称，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or f"chatbot_model_{timestamp}"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # 确保模型目录存在
    ensure_dir_exists(MODEL_DIR)
    
    # 准备训练数据
    all_patterns = []
    all_tags = []
    
    for intent in intents_data:
        tag = intent.get("tag", "")
        patterns = intent.get("patterns", [])
        
        for pattern in patterns:
            all_patterns.append(pattern)
            all_tags.append(tag)
    
    print(f"意图类别数: {len(intents_data)}")
    print(f"训练样本数: {len(all_patterns)}")
    
    # 创建文本处理器
    text_processor = TextProcessor(max_features=args.max_features)
    
    # 训练向量化器
    print("\n训练向量化器...")
    start_time = time.time()
    X = text_processor.fit_transform(all_patterns)
    print(f"向量化器训练完成，耗时 {time.time() - start_time:.2f} 秒")
    print(f"特征维度: {X.shape[1]}")
    
    # 创建标签映射
    unique_tags = sorted(list(set(all_tags)))
    tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
    y = np.array([tag_to_index[tag] for tag in all_tags])
    
    print(f"标签数量: {len(unique_tags)}")
    
    # 训练分类器
    print("\n训练分类器...")
    start_time = time.time()
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    
    # 训练随机森林分类器
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"分类器训练完成，耗时 {training_time:.2f} 秒")
    
    # 评估模型
    print("\n评估模型...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 保存模型
    model_data = {
        'model': model,
        'vectorizer': text_processor.vectorizer,
        'all_words': text_processor.get_vocabulary(),
        'tags': unique_tags
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"模型已保存至: {model_path}")
    return model_path

def evaluate_trained_model(model_path, training_file, args):
    """评估训练好的模型"""
    print("\n===== 开始评估模型 =====")
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return
    
    # 尝试加载模型评估模型性能
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('model')
        tags = model_data.get('tags')
        vectorizer = model_data.get('vectorizer')
        
        if not model or not tags or not vectorizer:
            print("模型数据不完整")
            return
        
        # 加载训练数据进行评估
        intents_data = load_training_data(training_file)
        if not intents_data:
            print("无法加载训练数据进行评估")
            return
        
        # 创建文本处理器并设置已训练好的向量化器
        text_processor = TextProcessor(max_features=args.max_features)
        text_processor.vectorizer = vectorizer
        
        # 计算每个意图的准确率
        tag_accuracies = {}
        for tag in tags:
            tag_patterns = []
            for intent in intents_data:
                if intent.get("tag") == tag:
                    tag_patterns.extend(intent.get("patterns", []))
            
            if tag_patterns:
                # 预测
                X = text_processor.transform(tag_patterns)
                predictions = model.predict(X)
                
                # 计算准确率
                correct = sum(1 for pred, pattern in zip(predictions, tag_patterns) 
                             if tags[pred] == tag)
                accuracy = correct / len(tag_patterns) if tag_patterns else 0
                tag_accuracies[tag] = accuracy
        
        # 整体准确率
        overall_accuracy = sum(tag_accuracies.values()) / len(tag_accuracies) if tag_accuracies else 0
        
        # 保存评估结果
        result = {
            "model_path": model_path,
            "training_file": training_file,
            "accuracy": overall_accuracy,
            "tag_accuracies": tag_accuracies,
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "max_features": args.max_features,
                "n_estimators": args.n_estimators,
                "augmentation_factor": args.factor if args.augment else 0
            }
        }
        
        # 打印详细结果
        print(f"\n总体准确率: {overall_accuracy:.4f}")
        print("\n各意图准确率:")
        for tag, acc in sorted(tag_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"{tag}: {acc:.4f}")
        
        # 保存评估结果到文件
        eval_file = os.path.join(os.path.dirname(model_path), f"{os.path.basename(model_path)}_evaluation.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存到: {eval_file}")
    
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 如果有测试脚本并且指定了测试，则运行测试
    if args.test and os.path.exists(os.path.join(root_dir, "scripts", "test_chatbot.py")):
        print("\n===== 开始测试模型对话 =====")
        test_cmd = [sys.executable, os.path.join(root_dir, "scripts", "test_chatbot.py"), "-m", model_path]
        if args.debug:
            test_cmd.append("--debug")
        
        print(f"执行测试命令: {' '.join(test_cmd)}")
        try:
            subprocess.run(test_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"测试过程中出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="训练数据扩展和模型训练一体化工具")
    
    # 数据处理参数
    parser.add_argument("--files", nargs="+", help="要处理的训练数据文件列表，如不指定则处理所有json文件")
    parser.add_argument("--output", default="merged_training_data.json", help="处理后的训练数据输出文件名")
    parser.add_argument("--augment", action="store_true", help="是否应用数据增强")
    parser.add_argument("--factor", type=int, default=3, help="数据增强因子，默认为3")
    
    # 模型训练参数
    parser.add_argument("--model-name", help="训练后的模型名称，默认使用时间戳")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF最大特征数，默认5000")
    parser.add_argument("--n-estimators", type=int, default=100, help="随机森林树的数量，默认100")
    
    # 评估和测试参数
    parser.add_argument("--skip-train", action="store_true", help="跳过训练步骤，只进行数据处理")
    parser.add_argument("--test", action="store_true", help="训练后进行测试对话")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 步骤1: 扩展训练数据
    training_file = extend_data(args)
    if not training_file:
        return
    
    # 如果指定跳过训练，则在此结束
    if args.skip_train:
        print("已跳过训练步骤，处理完成!")
        return
    
    # 步骤2: 训练模型
    model_path = train_model(training_file, args)
    if not model_path:
        return
    
    # 步骤3: 评估模型
    evaluate_trained_model(model_path, training_file, args)
    
    print("\n处理完成! 新模型已训练并评估，可用于聊天和部署。")

if __name__ == "__main__":
    main() 