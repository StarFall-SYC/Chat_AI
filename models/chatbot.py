import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import os
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
import time
from tqdm import tqdm
import sys
import traceback
from datetime import datetime
import argparse
import gc  # 导入垃圾回收模块

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from models.text_processor import TextProcessor
    from models.resource_manager import ResourceManager
except ImportError:
    # 本地调试时可能需要调整路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.text_processor import TextProcessor
    from models.resource_manager import ResourceManager

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        """
        高级聊天模型
        
        Args:
            input_size: 输入向量的维度
            hidden_size: 隐藏层维度
            output_size: 输出类别数
            dropout_rate: dropout比率，用于防止过拟合
        """
        super().__init__()
        
        # 定义三层神经网络，每层之间添加批归一化和dropout
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入向量 [batch_size, input_size]
            
        Returns:
            输出向量 [batch_size, output_size]
        """
        # 第一层: 输入 -> 隐藏层1
        x = self.fc1(x)
        if x.size(0) > 1:  # 只有当批大小大于1时才使用BatchNorm
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 第二层: 隐藏层1 -> 隐藏层2
        x = self.fc2(x)
        if x.size(0) > 1:  # 只有当批大小大于1时才使用BatchNorm
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 第三层: 隐藏层2 -> 输出层
        x = self.fc3(x)
        
        return x

class ChatbotManager:
    """
    聊天机器人管理器，负责加载模型、处理用户输入和生成回复
    """
    def __init__(self, model_path=None):
        """初始化聊天机器人管理器"""
        # 解析命令行参数
        parser = argparse.ArgumentParser(description="AI聊天机器人测试")
        parser.add_argument("-m", "--model", help="指定模型文件路径")
        parser.add_argument("--debug", action="store_true", help="启用调试模式")
        
        # 仅解析已知参数，忽略未知参数
        args, _ = parser.parse_known_args()
        
        # 获取资源管理器实例
        self.resource_manager = ResourceManager()
        
        # 路径设置
        self.base_path = self.resource_manager.base_path
        self.models_dir = self.resource_manager.models_dir
        self.training_data_path = os.path.join(self.base_path, 'data', 'training', 'intents.json')
        
        # 如果通过参数提供了模型路径，优先使用该路径
        self.custom_model_path = model_path or args.model
        
        # 初始化变量
        self.model = None
        self.vectorizer = None
        self.all_words = []
        self.tags = []
        self.text_processor = TextProcessor()
        self.training_data = []
        self.history = []
        
        # 调试模式设置
        self.debug_mode = args.debug or self.resource_manager.is_debug_mode()
        self.resource_manager.set_debug_mode(self.debug_mode)
        
        # 置信度阈值设置
        self.confidence_threshold = self.resource_manager.get_confidence_threshold()
        
        # 加载训练数据
        self.load_training_data()
        
        # 尝试加载模型
        self.load_model()
    
    def load_training_data(self):
        """加载训练数据"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                print(f"训练数据加载成功，包含 {len(self.training_data)} 个意图")
            else:
                print(f"训练数据文件不存在: {self.training_data_path}")
        except Exception as e:
            print(f"加载训练数据时出错: {str(e)}")
            traceback.print_exc()
      
    def load_model(self):
        """加载预训练模型"""
        try:
            # 确定模型路径
            if self.custom_model_path:
                # 如果提供了自定义路径，使用该路径
                model_path = self.custom_model_path
                if not os.path.isabs(model_path):
                    # 如果是相对路径，相对于项目根目录
                    model_path = os.path.join(self.base_path, model_path)
                print(f"使用指定的模型: {model_path}")
            else:
                # 否则使用默认路径
                model_path = os.path.join(self.models_dir, 'chat_model.pth')
            
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}")
                return False
            
            # 通过资源管理器加载模型
            model_id = os.path.basename(model_path)
            result = self.resource_manager.load_model(model_id)
            
            if not result["success"]:
                print(f"警告: 模型加载失败: {result['error']}")
                return False
            
            model_data = result["model"]
            
            self.model = model_data.get('model')
            self.all_words = model_data.get('all_words', [])
            self.tags = model_data.get('tags', [])
            self.vectorizer = model_data.get('vectorizer')
            
            # 加载相关训练数据
            self.load_matching_training_data()
            
            print(f"模型加载成功，包含 {len(self.tags)} 个意图")
            
            # 如果包含媒体资源，输出统计信息
            media_count = len(self.resource_manager.media_resources)
            if media_count > 0:
                print(f"加载了 {media_count} 个媒体资源引用")
            
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_matching_training_data(self):
        """尝试加载与当前模型匹配的训练数据"""
        if not self.tags:
            return
        
        # 搜索可能的训练数据文件
        training_dir = os.path.join(self.base_path, 'data', 'training')
        
        # 优先检查完整训练数据文件
        complete_data_file = os.path.join(training_dir, 'complete_training_data.json')
        if os.path.exists(complete_data_file):
            try:
                with open(complete_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                print(f"已加载完整训练数据: {complete_data_file}")
                return
            except Exception as e:
                print(f"加载完整训练数据时出错: {str(e)}")
        
        # 如果完整数据文件不存在，尝试其他可能的数据文件
        potential_files = [
            os.path.join(training_dir, 'merged_intents.json'),
            os.path.join(training_dir, 'extended_intents.json'),
            os.path.join(training_dir, 'advanced_intents.json'),
            os.path.join(training_dir, 'intents.json')
        ]
        
        # 查找包含当前意图标签的训练数据文件
        for file_path in potential_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 检查这个文件中是否包含我们需要的意图
                    tags_in_file = {item.get('tag') for item in data if isinstance(item, dict) and 'tag' in item}
                    model_tags = set(self.tags)
                    
                    # 如果文件中的标签覆盖了模型中的大部分标签（>50%），使用此文件
                    if len(model_tags.intersection(tags_in_file)) > len(model_tags) * 0.5:
                        self.training_data = data
                        print(f"已加载匹配的训练数据: {file_path}")
                        return
                except Exception as e:
                    print(f"尝试加载训练数据 {file_path} 时出错: {str(e)}")
        
        # 如果没有找到匹配的文件，创建中文默认回复
        if not self.training_data:
            # 创建带有默认回复的空意图
            self.training_data = []
            for tag in self.tags:
                # 为一些常见意图提供中文回复
                if tag == "greeting":
                    responses = ["你好！很高兴与你交流。", "嗨！有什么可以帮助你的吗？", "你好呀！今天过得怎么样？"]
                elif tag == "goodbye":
                    responses = ["再见！祝你有美好的一天。", "下次再聊！", "再见，随时欢迎回来聊天。"]
                elif tag == "thanks":
                    responses = ["不客气！", "很高兴能帮到你。", "随时为你服务！"]
                elif tag == "help":
                    responses = ["我可以回答各种问题，请告诉我你需要什么帮助？", "需要什么帮助吗？请告诉我具体问题。"]
                else:
                    # 其他意图使用通用回复
                    responses = [f"这是关于{tag}的回复。", f"我了解{tag}相关的内容，你想了解什么具体方面？"]
                
                self.training_data.append({
                    'tag': tag,
                    'patterns': [f"{tag} related query"],
                    'responses': responses
                })
            print("未找到匹配的训练数据文件，已创建中文默认回复")
    
    def get_response(self, text, include_media=True):
        """
        根据输入文本生成回复
        
        Args:
            text: 用户输入文本
            include_media: 是否包含媒体信息
            
        Returns:
            dict: 回复数据，包含文本和可能的媒体信息
        """
        # 存储回复
        reply = {"text": "", "media_type": None, "media_path": None}
        
        try:
            if not self.model:
                reply["text"] = "我还没有经过训练，无法回答问题。请先训练我！"
                return reply
                
            # 预处理输入文本
            tokens = self.text_processor.tokenize(text)
            
            # 向量化
            X = self.vectorizer.transform([text]).toarray()
            
            # 预测
            predictions = self.model.predict_proba(X)[0]
            predicted_index = predictions.argmax()
            confidence = predictions[predicted_index]
            
            predicted_tag = self.tags[predicted_index]
            
            if self.debug_mode:
                print(f"预测标签: {predicted_tag}, 置信度: {confidence:.4f}")
            
            # 如果置信度低于阈值，给出不确定的回复
            if confidence < self.confidence_threshold:
                reply["text"] = "对不起，我不太明白你的意思。能否换个方式表达？"
                return reply
            
            # 查找匹配的意图
            for intent in self.training_data:
                if intent['tag'] == predicted_tag:
                    # 随机选择一个回复
                    responses = intent.get('responses', ["我理解了。"])
                    reply["text"] = random.choice(responses)
                    
                    # 检查是否有媒体数据与该意图关联
                    if include_media:
                        media_info = self.resource_manager.get_media_resource(predicted_tag)
                        if media_info:
                            reply["media_type"] = media_info["type"]
                            reply["media_path"] = media_info["path"]
                    
                    break
            
            # 如果没有找到匹配的意图
            if not reply["text"]:
                reply["text"] = "对不起，我无法理解你的请求。"
                
            # 保存到历史记录
            self.history.append({
                'input': text,
                'response': reply,
                'timestamp': datetime.now().isoformat()
            })
            
            return reply
            
        except Exception as e:
            error_message = f"生成回复时出错: {str(e)}"
            print(error_message)
            traceback.print_exc()
            reply["text"] = "抱歉，处理您的请求时出错了。"
            return reply
    
    def add_to_history(self, user_message, bot_response):
        """添加对话到历史记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "timestamp": timestamp,
            "user": user_message,
            "bot": bot_response
        })
    
    def get_conversation_history(self):
        """获取对话历史"""
        # 将历史对话转换为更通用的格式
        conversation = []
        for item in self.history:
            conversation.append({
                "timestamp": item["timestamp"],
                "role": "user",
                "content": item["user"]
            })
            conversation.append({
                "timestamp": item["timestamp"],
                "role": "assistant",
                "content": item["bot"]
            })
        return conversation
    
    def get_history(self):
        """获取对话历史的原始格式"""
        return self.history
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.history = []
        return True
    
    def save_history(self, file_path=None):
        """
        保存对话历史到文件
        
        Args:
            file_path (str, optional): 保存路径，如果不提供则使用默认路径
            
        Returns:
            bool: 保存是否成功
        """
        if not file_path:
            os.makedirs(os.path.join(self.base_path, 'data', 'history'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.base_path, 'data', 'history', f'chat_history_{timestamp}.json')
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存对话历史时出错: {str(e)}")
            return False
    
    def set_debug_mode(self, enabled):
        """设置调试模式"""
        self.debug_mode = enabled
        print(f"调试模式已{'启用' if enabled else '禁用'}")
        return self.debug_mode
    
    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
            print(f"置信度阈值已设置为: {threshold:.2f}")
            return True
        else:
            print(f"无效的置信度阈值: {threshold}，必须在0-1之间")
            return False
    
    def get_confidence_threshold(self):
        """获取当前置信度阈值"""
        return self.confidence_threshold
    
    def is_model_loaded(self):
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        return self.model is not None and self.vectorizer is not None
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 包含模型信息的字典，如果模型未加载则返回None
        """
        if not self.is_model_loaded():
            return None
            
        info = {
            "intent_count": len(self.tags) if self.tags else 0,
            "vocabulary_size": len(self.all_words) if self.all_words else 0,
            "feature_count": self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
            "confidence_threshold": self.confidence_threshold,
            "debug_mode": self.debug_mode,
            "model_type": self.model.__class__.__name__ if self.model else "Unknown"
        }
        
        # 尝试获取模型特定信息
        try:
            if hasattr(self.model, "n_estimators"):
                info["n_estimators"] = self.model.n_estimators
            if hasattr(self.model, "max_depth"):
                info["max_depth"] = self.model.max_depth
        except:
            pass
            
        return info
        
    def train(self, training_data, epochs=200, learning_rate=0.001, batch_size=16):
        """
        训练聊天模型
        
        Args:
            training_data (list): 训练数据列表
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            batch_size (int): 批大小
            
        Returns:
            bool: 训练是否成功
        """
        # 获取训练锁
        training_lock = self.resource_manager.get_lock("training")
        if not training_lock.acquire(blocking=False):
            print("错误: 已有训练任务在进行中")
            return False
        
        try:
            from models.text_processor import TextProcessor
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            import pickle
            import time
            from sklearn.metrics import accuracy_score, classification_report
            
            print("=" * 50)
            print("开始训练聊天模型")
            print("=" * 50)
            
            if not training_data:
                print("错误: 训练数据为空")
                return False
                
            print(f"训练数据包含 {len(training_data)} 个意图")
            
            # 处理多模态数据 - 提取媒体信息并保存
            media_resources = {}
            media_count = 0
            
            for intent in training_data:
                tag = intent.get("tag", "")
                media_type = intent.get("media_type")
                media_path = intent.get("media_path")
                
                # 如果意图包含媒体信息，保存到媒体资源映射中
                if tag and media_type and media_path and os.path.exists(media_path):
                    media_resources[tag] = {
                        "type": media_type,
                        "path": media_path
                    }
                    # 注册到资源管理器
                    self.resource_manager.register_media_resource(tag, media_type, media_path)
                    media_count += 1
            
            if media_count > 0:
                print(f"处理了 {media_count} 个媒体资源引用")
            
            # 准备训练数据
            all_patterns = []
            all_tags = []
            
            for intent in training_data:
                tag = intent.get("tag", "")
                patterns = intent.get("patterns", [])
                
                for pattern in patterns:
                    all_patterns.append(pattern)
                    all_tags.append(tag)
            
            print(f"共 {len(all_patterns)} 个训练样本")
            
            # 创建文本处理器，使用3000个最重要的特征，这在意图分类任务中通常是合适的
            self.text_processor = TextProcessor(max_features=3000)
            
            # 标记和处理文本
            print("\n处理训练文本...")
            start_time = time.time()
            
            # 分批处理训练样本，避免内存峰值过高
            max_batch_size = 500  # 每批处理的最大样本数
            
            if len(all_patterns) > max_batch_size:
                print(f"样本数量较大，将分批处理以优化内存使用")
                processed_chunks = []
                
                for i in range(0, len(all_patterns), max_batch_size):
                    chunk_patterns = all_patterns[i:i + max_batch_size]
                    print(f"处理样本批次 {i//max_batch_size + 1}/{(len(all_patterns) + max_batch_size - 1)//max_batch_size}...")
                    
                    tokenized_patterns_chunk = [self.text_processor.tokenize(pattern) for pattern in chunk_patterns]
                    processed_chunks.extend(tokenized_patterns_chunk)
                    
                    # 手动触发垃圾回收
                    del tokenized_patterns_chunk
                    gc.collect()
                
                tokenized_patterns = processed_chunks
            else:
                tokenized_patterns = [self.text_processor.tokenize(pattern) for pattern in all_patterns]
            
            print(f"文本处理完成，耗时 {time.time() - start_time:.2f} 秒")
            
            # 手动触发垃圾回收
            gc.collect()
            
            # 训练向量化器
            print("\n训练向量化器...")
            start_time = time.time()
            X = self.text_processor.fit_transform(all_patterns)
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
            
            # 释放不再需要的大型数组，减少内存使用
            del X
            gc.collect()
            
            print(f"训练集大小: {X_train.shape[0]} 样本")
            print(f"测试集大小: {X_test.shape[0]} 样本")
            
            # 创建分类器模型
            n_estimators = min(150, max(50, int(len(all_patterns) / 3)))
            
            print(f"创建随机森林分类器 (树数量: {n_estimators})...")
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1  # 使用所有可用的CPU核心
            )
            
            # 训练模型
            print(f"开始训练模型...")
            model_start_time = time.time()
            
            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                self.model.fit(X_train, y_train)
                epoch_time = time.time() - epoch_start_time
                
                # 计算训练集和测试集的准确率
                train_accuracy = self.model.score(X_train, y_train)
                test_accuracy = self.model.score(X_test, y_test)
                
                # 由于随机森林不使用epochs，我们可以模拟一下进度输出
                if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
                    print(f"轮次 [{epoch}/{epochs}], 耗时: {epoch_time:.2f}秒, 损失: {1.0 - train_accuracy:.4f}, 准确率: {test_accuracy:.2%}")
            
            # 评估最终模型
            print("\n评估模型...")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"测试集准确率: {accuracy:.4f}")
            
            # 保存模型数据
            self.vectorizer = self.text_processor.vectorizer
            self.all_words = self.text_processor.get_vocabulary()
            self.tags = unique_tags
            
            # 保存模型到文件
            model_path = os.path.join(self.models_dir, 'chat_model.pth')
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'all_words': self.all_words,
                'tags': self.tags,
                'media_resources': media_resources  # 保存媒体资源映射
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"\n模型已保存至: {model_path}")
            print("训练完成!")
            
            # 最终垃圾回收
            gc.collect()
            
            return True
        
        except Exception as e:
            print(f"\n训练过程中出错: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # 释放训练锁
            training_lock.release() 