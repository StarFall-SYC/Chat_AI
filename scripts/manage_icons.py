#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图标管理脚本
用于管理应用程序图标
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加项目根目录到模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入图标生成函数
from ui.app_icon import generate_and_save_app_icons

def list_icons():
    """列出当前图标状态"""
    icons_dir = os.path.join(project_root, 'ui', 'icons')
    if not os.path.exists(icons_dir):
        print(f"图标目录不存在: {icons_dir}")
        return False
    
    print("\n当前图标状态:")
    icon_files = [
        'app_icon.png', 
        'app_icon_256.png', 
        'app_icon_128.png', 
        'app_icon_64.png', 
        'app_icon_48.png', 
        'app_icon_32.png', 
        'app_icon_16.png'
    ]
    
    has_custom_icons = False
    for icon_file in icon_files:
        icon_path = os.path.join(icons_dir, icon_file)
        if os.path.exists(icon_path):
            size = os.path.getsize(icon_path)
            print(f"  [✓] {icon_file} ({format_size(size)})")
            has_custom_icons = True
        else:
            print(f"  [✗] {icon_file} (不存在)")
    
    if has_custom_icons:
        print("\n您正在使用自定义图标。应用程序启动时将使用这些图标。")
    else:
        print("\n未找到任何图标。应用程序启动时将生成默认图标。")
    
    return True

def restore_default_icons():
    """恢复默认图标"""
    icons_dir = os.path.join(project_root, 'ui', 'icons')
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir, exist_ok=True)
    
    # 删除现有图标
    icon_files = [
        'app_icon.png', 
        'app_icon_256.png', 
        'app_icon_128.png', 
        'app_icon_64.png', 
        'app_icon_48.png', 
        'app_icon_32.png', 
        'app_icon_16.png'
    ]
    
    for icon_file in icon_files:
        icon_path = os.path.join(icons_dir, icon_file)
        if os.path.exists(icon_path):
            try:
                os.remove(icon_path)
                print(f"已删除图标: {icon_file}")
            except Exception as e:
                print(f"删除图标时出错 {icon_file}: {str(e)}")
    
    # 生成默认图标
    print("\n正在生成默认图标...")
    try:
        path = generate_and_save_app_icons()
        print(f"默认图标已生成: {path}")
        return True
    except Exception as e:
        print(f"生成默认图标时出错: {str(e)}")
        return False

def backup_icons():
    """备份当前图标"""
    icons_dir = os.path.join(project_root, 'ui', 'icons')
    if not os.path.exists(icons_dir):
        print(f"图标目录不存在: {icons_dir}")
        return False
    
    backup_dir = os.path.join(project_root, 'data', 'backup', 'icons')
    os.makedirs(backup_dir, exist_ok=True)
    
    # 复制图标文件
    icon_files = [
        'app_icon.png', 
        'app_icon_256.png', 
        'app_icon_128.png', 
        'app_icon_64.png', 
        'app_icon_48.png', 
        'app_icon_32.png', 
        'app_icon_16.png'
    ]
    
    has_backed_up = False
    for icon_file in icon_files:
        icon_path = os.path.join(icons_dir, icon_file)
        if os.path.exists(icon_path):
            try:
                shutil.copy2(icon_path, os.path.join(backup_dir, icon_file))
                print(f"已备份图标: {icon_file}")
                has_backed_up = True
            except Exception as e:
                print(f"备份图标时出错 {icon_file}: {str(e)}")
    
    if has_backed_up:
        print(f"\n图标已备份到: {backup_dir}")
        return True
    else:
        print("\n未找到任何图标，无法备份。")
        return False

def restore_from_backup():
    """从备份恢复图标"""
    backup_dir = os.path.join(project_root, 'data', 'backup', 'icons')
    if not os.path.exists(backup_dir):
        print(f"备份目录不存在: {backup_dir}")
        return False
    
    icons_dir = os.path.join(project_root, 'ui', 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    # 复制图标文件
    icon_files = [
        'app_icon.png', 
        'app_icon_256.png', 
        'app_icon_128.png', 
        'app_icon_64.png', 
        'app_icon_48.png', 
        'app_icon_32.png', 
        'app_icon_16.png'
    ]
    
    has_restored = False
    for icon_file in icon_files:
        backup_path = os.path.join(backup_dir, icon_file)
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, os.path.join(icons_dir, icon_file))
                print(f"已恢复图标: {icon_file}")
                has_restored = True
            except Exception as e:
                print(f"恢复图标时出错 {icon_file}: {str(e)}")
    
    if has_restored:
        print(f"\n图标已从备份恢复到: {icons_dir}")
        return True
    else:
        print("\n未找到任何备份图标，无法恢复。")
        return False

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} 字节"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="应用程序图标管理")
    parser.add_argument("-l", "--list", action="store_true", help="列出当前图标状态")
    parser.add_argument("-r", "--restore", action="store_true", help="恢复默认图标")
    parser.add_argument("-b", "--backup", action="store_true", help="备份当前图标")
    parser.add_argument("-f", "--from-backup", action="store_true", help="从备份恢复图标")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if not (args.list or args.restore or args.backup or args.from_backup):
        parser.print_help()
        list_icons()
        return 0
    
    if args.list:
        list_icons()
    
    if args.restore:
        restore_default_icons()
    
    if args.backup:
        backup_icons()
    
    if args.from_backup:
        restore_from_backup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 