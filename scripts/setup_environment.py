#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
环境设置脚本
用于初始化项目所需的目录和文件
"""

import os
import sys
import shutil
import argparse

def setup_environment(force=False):
    """
    设置项目环境
    
    Args:
        force: 是否强制重新创建目录
        
    Returns:
        bool: 是否成功
    """
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 要创建的目录列表
    directories = [
        os.path.join(base_dir, 'data', 'models'),               # 模型保存目录
        os.path.join(base_dir, 'data', 'training'),             # 训练数据目录
        os.path.join(base_dir, 'data', 'history'),              # 历史记录目录
        os.path.join(base_dir, 'data', 'generated', 'images'),  # 生成图像目录
        os.path.join(base_dir, 'data', 'generated', 'videos'),  # 生成视频目录
    ]
    
    created = []
    errors = []
    
    print("正在设置项目环境...")
    
    # 创建目录
    for directory in directories:
        try:
            # 如果目录存在且强制重新创建
            if os.path.exists(directory) and force:
                shutil.rmtree(directory)
                print(f"已删除并重新创建目录：{directory}")
            
            # 创建目录
            if not os.path.exists(directory):
                os.makedirs(directory)
                created.append(directory)
                print(f"已创建目录：{directory}")
            else:
                print(f"目录已存在：{directory}")
        except Exception as e:
            errors.append((directory, str(e)))
            print(f"创建目录失败：{directory}, 错误：{str(e)}")
    
    # 检查是否存在必要的文件
    required_files = [
        os.path.join(base_dir, 'data', 'training', 'training_data.json')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            # 如果训练数据文件不存在，创建一个空的
            try:
                # 获取目录
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # 写入初始数据
                if file_path.endswith('training_data.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('{"intents": []}')
                    print(f"已创建训练数据文件：{file_path}")
            except Exception as e:
                errors.append((file_path, str(e)))
                print(f"创建文件失败：{file_path}, 错误：{str(e)}")
    
    # 输出结果
    print("\n环境设置完成!")
    print(f"创建的目录数量：{len(created)}")
    if errors:
        print(f"出现的错误数量：{len(errors)}")
        for item, error in errors:
            print(f" - {item}: {error}")
    
    return len(errors) == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="设置项目环境")
    parser.add_argument("-f", "--force", action="store_true", help="强制重新创建目录")
    args = parser.parse_args()
    
    success = setup_environment(force=args.force)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 