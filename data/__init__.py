"""
数据目录
这个目录包含训练数据和模型加载/保存的功能
"""

import os

# 确保训练数据目录存在
training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")
if not os.path.exists(training_dir):
    os.makedirs(training_dir)
    print(f"已创建训练数据目录: {training_dir}")

# 数据模块导入
try:
    from .data_loader import load_training_data, save_training_data
except ImportError:
    # 定义备用函数，防止模块导入失败
    def load_training_data(file_path=None):
        """加载训练数据"""
        if file_path is None:
            # 默认路径
            file_path = os.path.join(os.path.dirname(__file__), "training/intents.json")
        
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['intents']
        except Exception as e:
            print(f"加载训练数据时出错: {e}")
            return []

    def save_training_data(data, file_path=None):
        """保存训练数据"""
        if file_path is None:
            # 默认路径
            file_path = os.path.join(os.path.dirname(__file__), "training/intents.json")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 构建数据格式
            import json
            formatted_data = {"intents": data}
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存训练数据时出错: {e}")
            return False 