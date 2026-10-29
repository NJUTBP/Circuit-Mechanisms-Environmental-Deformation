# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:12:05 2025

@author: 17534
"""

"""
放电野长度计算 - 简洁函数式接口

网格细胞：基于自相关中心峰
位置细胞：直接基于rate map
"""

import numpy as np
from scipy.ndimage import label, gaussian_filter
from scipy.signal import correlate2d
from typing import Tuple, Dict, Optional, List, Union
import opexebo


# ========== 网格细胞放电野长度 ==========

def compute_grid_field_length(
    rate_map: np.ndarray,
    bin_size: float = 2.0,
    threshold_ratio: float = 0.1
) -> Dict:
    """
    计算网格细胞放电野长度（基于自相关中心峰）
    
    方法：
    1. 计算rate map的二维自相关
    2. 找到中心峰（所有与中心连通且>10%最大值的像素）
    3. 分别计算X和Y方向的跨度
    
    参数:
        rate_map: 二维发放率图 (n_bins_y, n_bins_x)
        bin_size: 每个bin的大小 (cm)
        threshold_ratio: 阈值比例（默认0.1即10%）
        
    返回:
        dict: {
            'length_x': X方向长度 (cm),
            'length_y': Y方向长度 (cm),
            'aspect_ratio': 长宽比
        }
    """
    # 处理NaN
    rate_map_filled = rate_map.copy()
    rate_map_filled[np.isnan(rate_map_filled)] = 0
    
    # 计算自相关
    mean_rate = np.nanmean(rate_map)
    rate_map_centered = rate_map_filled - mean_rate
    autocorr = correlate2d(rate_map_centered, rate_map_centered, mode='full')
    
    autocorr = autocorr / autocorr.max()
    
    # 找到中心位置
    center_y, center_x = np.array(autocorr.shape) // 2
    
    # 阈值
    threshold = threshold_ratio
    
    # 找到所有大于阈值的像素
    above_threshold = autocorr > threshold
    
    # 标记连通区域
    labeled, n_features = label(above_threshold)
    
    # 找到包含中心点的区域
    center_label = labeled[center_y, center_x]
    
    if center_label == 0:
        return {
            'length_x': np.nan,
            'length_y': np.nan,
            'aspect_ratio': np.nan,
            'error': 'Center not in any region'
        }
    
    # 中心峰掩码
    central_peak_mask = labeled == center_label
    peak_coords = np.where(central_peak_mask)
    
    # 计算各维度的跨度
    y_min, y_max = peak_coords[0].min(), peak_coords[0].max()
    x_min, x_max = peak_coords[1].min(), peak_coords[1].max()
    
    length_y = (y_max - y_min + 1) * bin_size
    length_x = (x_max - x_min + 1) * bin_size
    
    return {
        'length_x': length_x,
        'length_y': length_y,
        'aspect_ratio': length_x / length_y if length_y > 0 else np.nan
    }


def compute_grid_field_length_batch(
    rate_maps: List[np.ndarray],
    bin_size: float = 2.0,
    threshold_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算多个网格细胞的放电野长度
    
    参数:
        rate_maps: rate map列表
        bin_size: bin大小 (cm)
        threshold_ratio: 阈值比例
        
    返回:
        lengths_x: (n_cells,) X方向长度数组
        lengths_y: (n_cells,) Y方向长度数组
    """
    n_cells = len(rate_maps)
    lengths_x = np.zeros(n_cells)
    lengths_y = np.zeros(n_cells)
    
    for i, rate_map in enumerate(rate_maps):
        result = compute_grid_field_length(rate_map, bin_size, threshold_ratio)
        lengths_x[i] = result['length_x']
        lengths_y[i] = result['length_y']
    
    return lengths_x, lengths_y


# ========== 位置细胞放电野长度 ==========

def compute_place_field_length(
    rate_map: np.ndarray,
    bin_size: float = 2.0,
    threshold_ratio: float = 0.1,
    select: str = 'largest'
) -> Dict:
    """
    计算位置细胞放电野长度（直接基于rate map）
    
    方法：
    1. 找到rate map中>10%最大值的所有像素
    2. 标记连通区域
    3. 选择目标放电野（最大的或所有的）
    4. 计算该区域在X和Y方向的跨度
    
    参数:
        rate_map: 二维发放率图 (n_bins_y, n_bins_x)
        bin_size: 每个bin的大小 (cm)
        threshold_ratio: 阈值比例（默认0.1即10%）
        select: 'largest' - 只返回最大放电野
                'all' - 返回所有放电野
        
    返回:
        dict: {
            'length_x': X方向长度 (cm),
            'length_y': Y方向长度 (cm),
            'center': (x, y) 放电野中心 (cm),
            'area': 面积 (cm²),
            'n_fields': 放电野数量
        }
    """
    # 处理NaN
    rate_map_valid = rate_map.copy()
    rate_map_valid[np.isnan(rate_map_valid)] = 0
    
    # 阈值
    max_rate = rate_map_valid.max()
    if max_rate == 0:
        return {
            'length_x': np.nan,
            'length_y': np.nan,
            'center': (np.nan, np.nan),
            'area': np.nan,
            'n_fields': 0,
            'error': 'No activity'
        }
    
    threshold = threshold_ratio * max_rate
    above_threshold = rate_map_valid > threshold
    
    # 标记连通区域
    labeled, n_features = label(above_threshold)
    
    if n_features == 0:
        return {
            'length_x': np.nan,
            'length_y': np.nan,
            'center': (np.nan, np.nan),
            'area': np.nan,
            'n_fields': 0,
            'error': 'No fields found'
        }
    
    # 分析每个放电野
    fields = []
    for field_id in range(1, n_features + 1):
        field_mask = labeled == field_id
        field_coords = np.where(field_mask)
        
        y_min, y_max = field_coords[0].min(), field_coords[0].max()
        x_min, x_max = field_coords[1].min(), field_coords[1].max()
        
        length_y = (y_max - y_min + 1) * bin_size
        length_x = (x_max - x_min + 1) * bin_size
        
        center_y = np.mean(field_coords[0]) * bin_size
        center_x = np.mean(field_coords[1]) * bin_size
        
        area = np.sum(field_mask) * bin_size ** 2
        
        fields.append({
            'length_x': length_x,
            'length_y': length_y,
            'center': (center_x, center_y),
            'area': area,
            'aspect_ratio': length_x / length_y if length_y > 0 else np.nan
        })
    
    if select == 'largest':
        selected = max(fields, key=lambda x: x['area'])
        selected['n_fields'] = n_features
        return selected
    else:  # 'all'
        return {
            'n_fields': n_features,
            'fields': fields
        }


def compute_place_field_length_batch(
    rate_maps: List[np.ndarray],
    bin_size: float = 2.0,
    threshold_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算多个位置细胞的放电野长度
    
    参数:
        rate_maps: rate map列表
        bin_size: bin大小 (cm)
        threshold_ratio: 阈值比例
        
    返回:
        lengths_x: (n_cells,) X方向长度数组
        lengths_y: (n_cells,) Y方向长度数组
    """
    n_cells = len(rate_maps)
    lengths_x = np.zeros(n_cells)
    lengths_y = np.zeros(n_cells)
    
    for i, rate_map in enumerate(rate_maps):
        result = compute_place_field_length(rate_map, bin_size, threshold_ratio)
        lengths_x[i] = result['length_x']
        lengths_y[i] = result['length_y']
    
    return lengths_x, lengths_y


# ========== 从原始数据计算rate map ==========

def compute_rate_map(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    bin_size: float = 2.0,
    sigma: float = 2.0
) -> np.ndarray:
    """
    从位置和spike数据计算rate map
    
    参数:
        positions: (n_samples, 2) 位置数据 [x, y]
        spike_train: (n_samples,) spike数据
        env_size: (width, height) 环境尺寸
        bin_size: bin大小 (cm)
        sigma: 高斯平滑sigma (cm)
        
    返回:
        rate_map: (n_bins_y, n_bins_x) 发放率图
    """
    n_bins_x = int(env_size[0] / bin_size)
    n_bins_y = int(env_size[1] / bin_size)
    
    bin_x = np.clip((positions[:, 0] / bin_size).astype(int), 0, n_bins_x - 1)
    bin_y = np.clip((positions[:, 1] / bin_size).astype(int), 0, n_bins_y - 1)
    
    spike_map = np.zeros((n_bins_y, n_bins_x))
    occ_map = np.zeros((n_bins_y, n_bins_x))
    
    for i in range(len(bin_x)):
        spike_map[bin_y[i], bin_x[i]] += spike_train[i]
        occ_map[bin_y[i], bin_x[i]] += 1
    
    # 高斯平滑
    sigma_bins = sigma / bin_size
    spike_map = gaussian_filter(spike_map, sigma_bins)
    occ_map = gaussian_filter(occ_map, sigma_bins)
    
    # 避免除零
    occ_map[occ_map < 0.1] = np.nan
    rate_map = spike_map / occ_map
    
    return rate_map


# ========== 便捷的一体化函数 ==========

def compute_grid_field_length_from_data(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    bin_size: float = 2.0,
    sigma: float = 2.0,
    threshold_ratio: float = 0.1
) -> Dict:
    """
    从原始数据直接计算网格细胞放电野长度
    
    参数:
        positions: (n_samples, 2) 位置数据
        spike_train: (n_samples,) spike数据
        env_size: 环境尺寸
        bin_size: bin大小
        sigma: 平滑参数
        threshold_ratio: 阈值比例
        
    返回:
        dict: 包含 length_x, length_y 等
    """
    rate_map = compute_rate_map(positions, spike_train, env_size, bin_size, sigma)
    return compute_grid_field_length(rate_map, bin_size, threshold_ratio)


def compute_place_field_length_from_data(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    bin_size: float = 2.0,
    sigma: float = 2.0,
    threshold_ratio: float = 0.1
) -> Dict:
    """
    从原始数据直接计算位置细胞放电野长度
    
    参数:
        positions: (n_samples, 2) 位置数据
        spike_train: (n_samples,) spike数据
        env_size: 环境尺寸
        bin_size: bin大小
        sigma: 平滑参数
        threshold_ratio: 阈值比例
        
    返回:
        dict: 包含 length_x, length_y, center, area 等
    """
    rate_map = compute_rate_map(positions, spike_train, env_size, bin_size, sigma)
    return compute_place_field_length(rate_map, bin_size, threshold_ratio)




# ========== 演示 ==========

def demo():
    """演示放电野长度计算"""
    print("=" * 60)
    print("放电野长度计算演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    env_size = (100, 100)
    bin_size = 2.0
    n_bins = int(env_size[0] / bin_size)
    
    # 创建坐标网格
    x = np.arange(n_bins) * bin_size
    y = np.arange(n_bins) * bin_size
    X, Y = np.meshgrid(x, y)
    
    # ========== 1. 网格细胞 ==========
    print("\n1. 网格细胞放电野长度")
    print("-" * 40)
    
    # 创建网格细胞rate map
    grid_scale = 30
    k = 4 * np.pi / (np.sqrt(3) * grid_scale)
    grid_rate = np.zeros_like(X)
    for theta in [0, np.pi/3, 2*np.pi/3]:
        grid_rate += np.cos(k * (X * np.cos(theta) + Y * np.sin(theta)))
    grid_rate = (grid_rate / 3 + 1) / 2 * 10
    
    result = compute_grid_field_length(grid_rate, bin_size)
    print(f"网格细胞放电野:")
    print(f"  X方向: {result['length_x']:.1f} cm")
    print(f"  Y方向: {result['length_y']:.1f} cm")
    print(f"  长宽比: {result['aspect_ratio']:.2f}")
    
    # ========== 2. 位置细胞 ==========
    print("\n2. 位置细胞放电野长度")
    print("-" * 40)
    
    # 创建位置细胞rate map（圆形）
    place_rate = 15 * np.exp(
        -((X - 60)**2 + (Y - 40)**2) / (2 * 12**2)
    )
    
    result = compute_place_field_length(place_rate, bin_size)
    print(f"圆形放电野:")
    print(f"  X方向: {result['length_x']:.1f} cm")
    print(f"  Y方向: {result['length_y']:.1f} cm")
    print(f"  中心: ({result['center'][0]:.1f}, {result['center'][1]:.1f}) cm")
    print(f"  面积: {result['area']:.1f} cm²")
    
    # 创建长条形放电野
    boundary_rate = 12 * np.exp(
        -((X - 50)**2 / (2 * 25**2) + (Y - 10)**2 / (2 * 5**2))
    )
    
    result = compute_place_field_length(boundary_rate, bin_size)
    print(f"\n长条形放电野:")
    print(f"  X方向: {result['length_x']:.1f} cm")
    print(f"  Y方向: {result['length_y']:.1f} cm")
    print(f"  长宽比: {result['aspect_ratio']:.2f}")
    
    # ========== 3. 批量计算 ==========
    print("\n3. 批量计算示例")
    print("-" * 40)
    
    # 生成多个rate maps
    grid_maps = []
    for i in range(10):
        phase = np.random.rand() * 2 * np.pi
        rate = np.zeros_like(X)
        for theta in [0, np.pi/3, 2*np.pi/3]:
            rate += np.cos(k * (X * np.cos(theta) + Y * np.sin(theta)) + phase)
        rate = (rate / 3 + 1) / 2 * 10
        grid_maps.append(rate)
    
    lengths_x, lengths_y = compute_grid_field_length_batch(grid_maps, bin_size)
    print(f"10个网格细胞:")
    print(f"  X方向平均: {np.mean(lengths_x):.1f} ± {np.std(lengths_x):.1f} cm")
    print(f"  Y方向平均: {np.mean(lengths_y):.1f} ± {np.std(lengths_y):.1f} cm")
    
    print("\n演示完成!")


if __name__ == "__main__":
    demo()