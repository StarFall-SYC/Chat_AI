"""
锁管理模块
提供线程安全的锁管理功能，控制并发访问
"""

import threading
from typing import Dict, Any

from .logger import app_logger


class LockManager:
    """锁管理器，提供对不同资源的并发访问控制"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LockManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化锁管理器"""
        if getattr(self, "_initialized", False):
            return
            
        # 全局资源锁
        self.global_locks = {
            "image_generator": threading.RLock(),
            "video_generator": threading.RLock(),
            "training": threading.RLock()
        }
        
        # 模型特定锁
        self.model_locks: Dict[str, threading.RLock] = {}
        
        # 操作锁，用于临界区保护
        self.operation_locks: Dict[str, threading.RLock] = {}
        
        # 活跃操作计数
        self.active_operations: Dict[str, int] = {}
        
        # 已获取锁的线程ID
        self.lock_owners: Dict[str, int] = {}
        
        self._initialized = True
    
    def get_lock(self, resource_id: str, category: str = "model") -> threading.RLock:
        """获取资源锁
        
        Args:
            resource_id: 资源ID
            category: 锁类别（"global", "model", "operation"）
            
        Returns:
            threading.RLock: 资源锁
        """
        if category == "global":
            if resource_id in self.global_locks:
                return self.global_locks[resource_id]
            else:
                app_logger.warning(f"请求不存在的全局锁: {resource_id}，将创建新锁")
                self.global_locks[resource_id] = threading.RLock()
                return self.global_locks[resource_id]
                
        elif category == "model":
            with self._lock:
                if resource_id not in self.model_locks:
                    self.model_locks[resource_id] = threading.RLock()
                return self.model_locks[resource_id]
                
        elif category == "operation":
            with self._lock:
                if resource_id not in self.operation_locks:
                    self.operation_locks[resource_id] = threading.RLock()
                return self.operation_locks[resource_id]
                
        else:
            app_logger.warning(f"未知的锁类别: {category}，将使用模型锁")
            return self.get_lock(resource_id, "model")
    
    def acquire_lock(self, resource_id: str, category: str = "model", timeout: float = -1) -> bool:
        """获取资源锁
        
        Args:
            resource_id: 资源ID
            category: 锁类别（"global", "model", "operation"）
            timeout: 超时时间（秒），-1表示无限等待
            
        Returns:
            bool: 是否成功获取锁
        """
        lock = self.get_lock(resource_id, category)
        success = lock.acquire(timeout=timeout if timeout > 0 else None)
        
        if success:
            thread_id = threading.get_ident()
            with self._lock:
                lock_key = f"{category}:{resource_id}"
                self.lock_owners[lock_key] = thread_id
                self.active_operations[lock_key] = self.active_operations.get(lock_key, 0) + 1
            app_logger.debug(f"线程 {thread_id} 获取了锁 {category}:{resource_id}")
        else:
            app_logger.warning(f"获取锁 {category}:{resource_id} 超时")
            
        return success
    
    def release_lock(self, resource_id: str, category: str = "model") -> bool:
        """释放资源锁
        
        Args:
            resource_id: 资源ID
            category: 锁类别（"global", "model", "operation"）
            
        Returns:
            bool: 是否成功释放锁
        """
        lock_key = f"{category}:{resource_id}"
        
        # 检查当前线程是否拥有锁
        thread_id = threading.get_ident()
        owner_id = self.lock_owners.get(lock_key)
        
        if owner_id != thread_id:
            app_logger.warning(f"线程 {thread_id} 尝试释放未持有的锁 {lock_key}，持有者: {owner_id}")
            return False
        
        lock = None
        if category == "global" and resource_id in self.global_locks:
            lock = self.global_locks[resource_id]
        elif category == "model" and resource_id in self.model_locks:
            lock = self.model_locks[resource_id]
        elif category == "operation" and resource_id in self.operation_locks:
            lock = self.operation_locks[resource_id]
        
        if lock:
            try:
                lock.release()
                with self._lock:
                    self.active_operations[lock_key] = max(0, self.active_operations.get(lock_key, 1) - 1)
                    if self.active_operations[lock_key] == 0:
                        self.lock_owners.pop(lock_key, None)
                app_logger.debug(f"线程 {thread_id} 释放了锁 {lock_key}")
                return True
            except RuntimeError:
                app_logger.error(f"释放锁 {lock_key} 失败，可能锁未被获取")
                return False
        else:
            app_logger.warning(f"尝试释放不存在的锁: {lock_key}")
            return False
    
    def is_locked(self, resource_id: str, category: str = "model") -> bool:
        """检查资源是否被锁定
        
        Args:
            resource_id: 资源ID
            category: 锁类别（"global", "model", "operation"）
            
        Returns:
            bool: 资源是否被锁定
        """
        lock_key = f"{category}:{resource_id}"
        return lock_key in self.lock_owners
    
    def get_active_operations(self) -> Dict[str, int]:
        """获取活跃操作计数
        
        Returns:
            Dict[str, int]: 活跃操作计数
        """
        return self.active_operations.copy()
    
    def clear_expired_locks(self):
        """清理过期的锁
        
        应在应用程序退出前调用，确保所有锁都被正确释放
        """
        for category in ["global", "model", "operation"]:
            locks = getattr(self, f"{category}_locks", {})
            for resource_id in list(locks.keys()):
                lock_key = f"{category}:{resource_id}"
                if lock_key in self.lock_owners:
                    app_logger.warning(f"强制释放锁: {lock_key}")
                    try:
                        self.release_lock(resource_id, category)
                    except Exception as e:
                        app_logger.error(f"释放锁 {lock_key} 时出错: {str(e)}")


# 创建锁管理器实例
lock_manager = LockManager()


class ResourceLock:
    """资源锁上下文管理器
    
    用于在with语句中自动获取和释放锁
    """
    
    def __init__(self, resource_id: str, category: str = "model", timeout: float = -1):
        """初始化资源锁
        
        Args:
            resource_id: 资源ID
            category: 锁类别（"global", "model", "operation"）
            timeout: 超时时间（秒），-1表示无限等待
        """
        self.resource_id = resource_id
        self.category = category
        self.timeout = timeout
        self.acquired = False
    
    def __enter__(self):
        """进入上下文，获取锁"""
        self.acquired = lock_manager.acquire_lock(
            self.resource_id, 
            self.category, 
            self.timeout
        )
        if not self.acquired:
            raise TimeoutError(f"获取锁 {self.category}:{self.resource_id} 超时")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，释放锁"""
        if self.acquired:
            lock_manager.release_lock(self.resource_id, self.category)
            self.acquired = False 