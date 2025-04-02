import os
import json
import time
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from models.text_processor import TextProcessor
from models.chatbot import ChatbotManager

def train_model(training_data, epochs=200, learning_rate=0.001, batch_size=32):
    """
    训练聊天机器人模型
    
    Args:
        training_data: 训练数据列表
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批处理大小
    
    Returns:
        bool: 训练是否成功
    """
    try:
        print("开始训练模型...")
        print(f"训练数据包含 {len(training_data)} 个意图")
        
        # 创建模型管理器
        chatbot = ChatbotManager()
        
        # 开始训练
        success = chatbot.train(
            training_data=training_data,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        if success:
            print("模型训练完成！")
            return True
        else:
            print("模型训练失败！")
            return False
            
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extend_training_data(base_data, new_data):
    """
    扩展训练数据
    
    Args:
        base_data: 基础训练数据
        new_data: 新的训练数据
    
    Returns:
        list: 合并后的训练数据
    """
    try:
        # 创建意图标签到数据的映射
        intent_map = {intent["tag"]: intent for intent in base_data}
        
        # 合并新数据
        for intent in new_data:
            tag = intent["tag"]
            if tag in intent_map:
                # 如果意图已存在，合并patterns和responses
                intent_map[tag]["patterns"].extend(intent["patterns"])
                intent_map[tag]["responses"].extend(intent["responses"])
                # 去重
                intent_map[tag]["patterns"] = list(set(intent_map[tag]["patterns"]))
                intent_map[tag]["responses"] = list(set(intent_map[tag]["responses"]))
            else:
                # 如果是新意图，直接添加
                intent_map[tag] = intent
        
        # 转换回列表
        merged_data = list(intent_map.values())
        
        print(f"数据合并完成！")
        print(f"原始意图数量: {len(base_data)}")
        print(f"新增意图数量: {len(new_data)}")
        print(f"合并后意图数量: {len(merged_data)}")
        
        return merged_data
        
    except Exception as e:
        print(f"合并训练数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return base_data

def main():
    """主函数"""
    # 加载训练数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    training_data_path = os.path.join(data_dir, "training_data.json")
    
    try:
        with open(training_data_path, "r", encoding="utf-8") as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"加载训练数据失败: {str(e)}")
        return
    
    # 训练参数
    epochs = 200
    learning_rate = 0.001
    batch_size = 32
    
    # 开始训练
    success = train_model(
        training_data=training_data,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    if success:
        print("训练成功完成！")
    else:
        print("训练失败！")

if __name__ == "__main__":
    main() 