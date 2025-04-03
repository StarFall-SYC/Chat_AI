#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
锁管理器测试脚本
测试锁管理器在多线程环境下的并发控制功能
"""

import os
import sys
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入锁管理器
from models.lock_manager import lock_manager, ResourceLock
from models.logger import app_logger


def worker_function(worker_id, resource_id, category, duration):
    """工作线程函数，尝试获取锁并持有一段时间
    
    Args:
        worker_id: 工作线程ID
        resource_id: 资源ID
        category: 锁类别
        duration: 持有锁的时间（秒）
    """
    thread_id = threading.get_ident()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} (ID: {thread_id}) 尝试获取锁 {category}:{resource_id}")
    
    try:
        # 使用上下文管理器获取锁
        with ResourceLock(resource_id, category, timeout=2):
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} (ID: {thread_id}) 已获取锁 {category}:{resource_id}")
            
            # 模拟工作
            time.sleep(duration)
            
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} (ID: {thread_id}) 完成工作，即将释放锁 {category}:{resource_id}")
    except TimeoutError:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} (ID: {thread_id}) 获取锁 {category}:{resource_id} 超时")


def test_concurrent_locking():
    """测试并发锁定"""
    print("\n===== 测试并发锁定 =====")
    
    # 创建线程池
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交多个任务，它们将竞争同一个资源
        futures = []
        for i in range(5):
            futures.append(executor.submit(
                worker_function, 
                f"worker-{i+1}", 
                "shared_resource", 
                "model", 
                random.uniform(0.5, 1.5)
            ))
        
        # 等待所有任务完成
        for future in futures:
            future.result()
    
    print("测试并发锁定完成")


def test_nested_locks():
    """测试嵌套锁"""
    print("\n===== 测试嵌套锁 =====")
    
    def nested_worker(worker_id):
        thread_id = threading.get_ident()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} (ID: {thread_id}) 开始获取嵌套锁")
        
        try:
            # 获取外层锁
            with ResourceLock("outer_resource", "global", timeout=1):
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} 获取了外层锁")
                
                # 模拟一些工作
                time.sleep(0.2)
                
                # 获取内层锁
                with ResourceLock("inner_resource", "model", timeout=1):
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} 获取了内层锁")
                    
                    # 模拟一些工作
                    time.sleep(0.5)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} 即将释放内层锁")
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} 已释放内层锁，即将释放外层锁")
        except TimeoutError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 {worker_id} 获取锁超时: {str(e)}")
    
    # 创建两个线程测试嵌套锁
    t1 = threading.Thread(target=nested_worker, args=("nested-1",))
    t2 = threading.Thread(target=nested_worker, args=("nested-2",))
    
    t1.start()
    time.sleep(0.1)  # 让第一个线程先启动
    t2.start()
    
    t1.join()
    t2.join()
    
    print("测试嵌套锁完成")


def test_deadlock_prevention():
    """测试死锁预防"""
    print("\n===== 测试死锁预防 =====")
    
    def worker_a():
        thread_id = threading.get_ident()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A (ID: {thread_id}) 尝试获取锁 resource_1")
        
        try:
            # 获取第一个资源
            with ResourceLock("resource_1", "model", timeout=1):
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 获取了锁 resource_1")
                
                # 模拟一些工作
                time.sleep(0.5)
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 尝试获取锁 resource_2")
                
                # 获取第二个资源（可能会超时，因为线程B已经持有）
                with ResourceLock("resource_2", "model", timeout=1):
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 获取了锁 resource_2")
                    
                    # 模拟一些工作
                    time.sleep(0.2)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 即将释放锁 resource_2")
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 已释放锁 resource_2，即将释放锁 resource_1")
        except TimeoutError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 A 获取锁超时: {str(e)}")
    
    def worker_b():
        thread_id = threading.get_ident()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B (ID: {thread_id}) 尝试获取锁 resource_2")
        
        try:
            # 获取第二个资源
            with ResourceLock("resource_2", "model", timeout=1):
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 获取了锁 resource_2")
                
                # 模拟一些工作
                time.sleep(0.5)
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 尝试获取锁 resource_1")
                
                # 获取第一个资源（可能会超时，因为线程A已经持有）
                with ResourceLock("resource_1", "model", timeout=1):
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 获取了锁 resource_1")
                    
                    # 模拟一些工作
                    time.sleep(0.2)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 即将释放锁 resource_1")
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 已释放锁 resource_1，即将释放锁 resource_2")
        except TimeoutError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 线程 B 获取锁超时: {str(e)}")
    
    # 创建两个线程，它们会尝试以相反的顺序获取两个锁
    t1 = threading.Thread(target=worker_a)
    t2 = threading.Thread(target=worker_b)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    print("测试死锁预防完成")


def main():
    """主函数"""
    print(f"锁管理器测试开始于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试并发锁定
    test_concurrent_locking()
    
    # 测试嵌套锁
    test_nested_locks()
    
    # 测试死锁预防
    test_deadlock_prevention()
    
    # 显示锁管理器状态
    active_operations = lock_manager.get_active_operations()
    print("\n当前活跃操作:")
    for lock_key, count in active_operations.items():
        print(f"  {lock_key}: {count}")
    
    # 清理锁
    print("\n清理所有锁...")
    lock_manager.clear_expired_locks()
    
    print(f"\n锁管理器测试完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 