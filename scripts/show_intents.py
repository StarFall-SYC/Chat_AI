#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
显示训练数据中的意图和统计信息
"""

import argparse
import json
import os
import sys

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='显示训练数据中的意图和统计信息')
    parser.add_argument('--files', nargs='+', help='要分析的训练数据文件')
    
    args = parser.parse_args()
    
    if not args.files:
        print("错误: 请提供至少一个训练数据文件")
        return 1
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            continue
        
        try:
            # 加载训练数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n文件: {file_path}")
            print("=" * 50)
            
            # 处理不同格式的训练数据
            intents = []
            if isinstance(data, dict) and "intents" in data:
                # 如果是 {"intents": [...]} 格式
                intents = data["intents"]
                print(f"数据格式: 包含 'intents' 键的字典")
            else:
                # 如果直接是意图列表
                intents = data
                print(f"数据格式: 意图列表")
            
            # 统计信息
            total_patterns = 0
            total_responses = 0
            
            print(f"\n共找到 {len(intents)} 个意图:")
            print("-" * 50)
            
            # 输出每个意图的信息
            for i, intent in enumerate(intents, 1):
                if not isinstance(intent, dict) or 'tag' not in intent:
                    print(f"警告: 第 {i} 个项目不是有效的意图格式")
                    continue
                
                tag = intent.get('tag', 'unknown')
                patterns = intent.get('patterns', [])
                responses = intent.get('responses', [])
                
                print(f"{i}. {tag} - {len(patterns)} 个模式, {len(responses)} 个响应")
                
                total_patterns += len(patterns)
                total_responses += len(responses)
            
            print("\n统计信息:")
            print("-" * 50)
            print(f"总意图数: {len(intents)}")
            print(f"总模式数: {total_patterns}")
            print(f"总响应数: {total_responses}")
            
            if len(intents) > 0:
                print(f"平均每个意图的模式数: {total_patterns / len(intents):.2f}")
                print(f"平均每个意图的响应数: {total_responses / len(intents):.2f}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 