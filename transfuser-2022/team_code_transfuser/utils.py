"""
工具函数模块

功能描述:
    提供坐标变换和角度归一化等工具函数
    主要用于LiDAR和车辆坐标系之间的转换
    
主要函数:
    - get_virtual_lidar_to_vehicle_transform(): 虚拟LiDAR到车辆坐标变换
    - get_lidar_to_vehicle_transform(): LiDAR到车辆坐标变换
    - get_lidar_to_bevimage_transform(): LiDAR到BEV图像坐标变换
    - normalize_angle(): 角度归一化到[-π, π]
    - normalize_angle_degree(): 角度归一化到[-180, 180]
"""

import numpy as np

def get_virtual_lidar_to_vehicle_transform():
    """
    获取虚拟LiDAR到车辆坐标系的变换矩阵
    
    说明:
        这是一个假的LiDAR坐标系，用于测试和可视化
        变换矩阵包含平移但不包含旋转
    
    返回:
        np.ndarray: 4x4的齐次变换矩阵
            - 平移: x=1.3m, y=0.0m, z=2.5m
            - 旋转: 无旋转(单位矩阵)
    
    示例:
        >>> T = get_virtual_lidar_to_vehicle_transform()
        >>> print(T.shape)
        (4, 4)
    """
    # 创建4x4单位矩阵
    T = np.eye(4)
    # 设置平移向量 (LiDAR安装位置)
    T[0, 3] = 1.3  # x方向: 车辆前方1.3米
    T[1, 3] = 0.0  # y方向: 车辆中心线
    T[2, 3] = 2.5  # z方向: 离地2.5米
    return T
        
def get_vehicle_to_virtual_lidar_transform():
    """
    获取车辆到虚拟LiDAR坐标系的变换矩阵
    
    说明:
        这是get_virtual_lidar_to_vehicle_transform()的逆变换
    
    返回:
        np.ndarray: 4x4的逆变换矩阵
    """
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_lidar_to_vehicle_transform():
    """
    获取真实LiDAR到车辆坐标系的变换矩阵
    
    说明:
        包含旋转和平移两部分:
        - 旋转: 90度绕z轴旋转 (LiDAR朝向调整)
        - 平移: 与虚拟LiDAR相同的安装位置
    
    返回:
        np.ndarray: 4x4的齐次变换矩阵
            旋转矩阵: [[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]]
            平移向量: [1.3, 0.0, 2.5]
    
    注意:
        旋转矩阵将LiDAR的x轴映射到车辆的y轴
        这是因为LiDAR和车辆的坐标系定义不同
    """
    # 定义旋转矩阵 (90度绕z轴)
    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=np.float32)
    # 创建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = rot  # 设置旋转部分

    # 设置平移部分 (LiDAR安装位置)
    T[0, 3] = 1.3  # x方向
    T[1, 3] = 0.0  # y方向
    T[2, 3] = 2.5  # z方向
    return T

def get_vehicle_to_lidar_transform():
    """
    获取车辆到真实LiDAR坐标系的变换矩阵
    
    说明:
        这是get_lidar_to_vehicle_transform()的逆变换
    
    返回:
        np.ndarray: 4x4的逆变换矩阵
    """
    return np.linalg.inv(get_lidar_to_vehicle_transform())

def get_lidar_to_bevimage_transform():
    """
    获取LiDAR坐标到BEV(鸟瞰图)图像坐标的变换矩阵
    
    说明:
        将3D LiDAR点云投影到2D BEV图像
        包含旋转、平移和缩放三个步骤
    
    返回:
        np.ndarray: 3x3的2D变换矩阵
            - 旋转: 90度
            - 平移: x方向16像素, y方向32像素
            - 缩放: 8像素/米 (分辨率)
    
    坐标系说明:
        - LiDAR: 以车辆为中心, x向前, y向左
        - BEV图像: 以图像左上角为原点, x向右, y向下
    
    示例:
        >>> T = get_lidar_to_bevimage_transform()
        >>> lidar_point = np.array([2.0, 1.0, 1.0])  # x=2m, y=1m
        >>> bev_point = T @ lidar_point
        >>> print(bev_point[:2])  # BEV图像坐标
    """
    # 定义旋转和平移
    T = np.array([[0, -1, 16],   # x_bev = -y_lidar + 16
                  [-1, 0, 32],   # y_bev = -x_lidar + 32
                  [0, 0, 1]], dtype=np.float32)
    # 应用缩放 (8像素/米)
    T[:2, :] *= 8

    return T

def normalize_angle(x):
    """
    将角度归一化到[-π, π]范围
    
    参数:
        x (float): 输入角度(弧度)
    
    返回:
        float: 归一化后的角度, 范围[-π, π]
    
    算法:
        1. 先将角度映射到[0, 2π)
        2. 如果大于π, 减去2π得到[-π, π)
    
    示例:
        >>> normalize_angle(3 * np.pi)
        -3.141592653589793  # 约等于-π
        >>> normalize_angle(np.pi / 2)
        1.5707963267948966  # 约等于π/2
    """
    x = x % (2 * np.pi)    # 强制映射到[0, 2π)
    if x > np.pi:          # 如果大于π, 移到[-π, π)
        x -= 2 * np.pi
    return x

def normalize_angle_degree(x):
    """
    将角度归一化到[-180, 180]范围
    
    参数:
        x (float): 输入角度(度)
    
    返回:
        float: 归一化后的角度, 范围[-180, 180]
    
    算法:
        1. 先将角度映射到[0, 360)
        2. 如果大于180, 减去360得到[-180, 180)
    
    示例:
        >>> normalize_angle_degree(270)
        -90.0
        >>> normalize_angle_degree(45)
        45.0
    """
    x = x % 360.0          # 强制映射到[0, 360)
    if (x > 180.0):        # 如果大于180, 移到[-180, 180)
        x -= 360.0
    return x
