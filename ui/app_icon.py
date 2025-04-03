"""
应用程序图标模块
提供应用程序图标生成功能
"""

from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, QPainterPath
from PyQt5.QtCore import Qt, QSize, QRect, QPoint, QPointF

import os


def generate_app_icon():
    """生成应用程序图标
    
    首先检查用户是否有自定义图标，如有则使用；
    否则，使用程序代码创建的矢量图标
    
    Returns:
        QIcon: 应用程序图标
    """
    # 检查图标目录
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    # 检查是否存在自定义图标
    main_icon_path = os.path.join(icons_dir, 'app_icon.png')
    custom_icon_exists = os.path.exists(main_icon_path)
    
    # 如果存在自定义图标，直接使用
    if custom_icon_exists:
        icon = QIcon(main_icon_path)
        # 检查其他尺寸图标是否存在
        for size in [16, 32, 48, 64, 128, 256]:
            size_icon_path = os.path.join(icons_dir, f'app_icon_{size}.png')
            if os.path.exists(size_icon_path):
                icon.addPixmap(QPixmap(size_icon_path))
        return icon
    
    # 如果不存在自定义图标，则创建默认图标
    icon = QIcon()
    
    for size in [16, 32, 48, 64, 128, 256]:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter()
        painter.begin(pixmap)
        
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        draw_app_icon(painter, QRect(0, 0, size, size))
        
        painter.end()
        
        # 添加到图标
        icon.addPixmap(pixmap)
    
    return icon


def draw_app_icon(painter, rect):
    """绘制应用程序图标
    
    Args:
        painter: QPainter对象
        rect: 绘制的矩形区域
    """
    # 获取图标的缩放比例
    scale = rect.width() / 256.0  # 256作为基准尺寸
    
    # 绘制圆形背景
    center = rect.center()
    radius = int(rect.width() * 0.45)
    
    # 绘制渐变背景
    gradient = QLinearGradient(
        QPointF(rect.left(), rect.top()),
        QPointF(rect.right(), rect.bottom())
    )
    gradient.setColorAt(0, QColor(33, 150, 243))  # 蓝色
    gradient.setColorAt(1, QColor(0, 188, 212))   # 青色
    
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(gradient))
    painter.drawEllipse(center, radius, radius)
    
    # 绘制AI文字
    font = QFont("Arial", int(60 * scale), QFont.Bold)
    painter.setFont(font)
    painter.setPen(QPen(Qt.white, 2 * scale))
    
    text_rect = QRect(
        rect.left(), 
        rect.top() + int(rect.height() * 0.25), 
        rect.width(), 
        int(rect.height() * 0.5)  # 确保使用整数值
    )
    painter.drawText(text_rect, Qt.AlignCenter, "AI")
    
    # 绘制对话气泡图标
    bubble_path = QPainterPath()
    
    # 第一个气泡
    bubble1_center = QPointF(
        center.x() - int(30 * scale),
        center.y() + int(20 * scale)
    )
    bubble1_radius = int(25 * scale)
    bubble_path.addEllipse(bubble1_center, bubble1_radius, bubble1_radius)
    
    # 第二个气泡
    bubble2_center = QPointF(
        center.x() + int(30 * scale),
        center.y() + int(20 * scale)
    )
    bubble2_radius = int(20 * scale)
    bubble_path.addEllipse(bubble2_center, bubble2_radius, bubble2_radius)
    
    # 绘制气泡
    painter.setPen(QPen(Qt.white, 2 * scale))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(bubble_path)
    
    # 绘制外圈
    outer_radius = int(rect.width() * 0.48)
    painter.setPen(QPen(QColor(255, 255, 255, 100), 3 * scale))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(center, outer_radius, outer_radius)


def generate_and_save_app_icons():
    """生成并保存应用程序图标
    
    生成多个尺寸的图标并保存到ui/icons目录
    如果图标文件已存在，则不覆盖
    """
    # 创建图标目录
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    # 生成并保存各种尺寸的图标
    main_icon_path = os.path.join(icons_dir, 'app_icon.png')
    
    # 检查主图标是否存在，如果已存在则不覆盖
    if os.path.exists(main_icon_path):
        print(f"自定义图标已存在: {main_icon_path}")
        return main_icon_path
    
    for size in [16, 32, 48, 64, 128, 256]:
        icon_path = os.path.join(icons_dir, f'app_icon_{size}.png')
        
        # 检查该尺寸的图标是否存在，若存在则跳过
        if os.path.exists(icon_path):
            print(f"自定义图标已存在: {icon_path}")
            continue
            
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        # 创建新的painter对象
        painter = QPainter()
        painter.begin(pixmap)
        
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # 使用整数矩形
        draw_app_icon(painter, QRect(0, 0, size, size))
        
        painter.end()
        
        # 保存图标
        pixmap.save(icon_path, 'PNG')
        
        # 将256尺寸的图标作为主图标
        if size == 256 and not os.path.exists(main_icon_path):
            pixmap.save(main_icon_path, 'PNG')
    
    return main_icon_path 