# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:32:53 2025

@author: 17534
"""

"""
网格细胞相位差计算 - 简洁函数式接口

用于批量计算多个细胞、多个环境、多个轴向的相位差
"""

import numpy as np
from scipy.ndimage import label
from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
from scipy import ndimage

def compute_phase_shift(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    grid_scale: float,
    axis: str = 'x',
    bin_size: float = 2.0,
    boundary_threshold: float = 12.0,
    n_bootstrap: int = 100,
    threshold_ratio: float = 0.1
) -> Dict:
    """
    计算单个细胞在指定轴向的相位差
    
    参数:
        positions: (n_samples, 2) 动物位置 [x, y]
        spike_train: (n_samples,) 每个时间bin的spike数
        env_size: (width, height) 环境尺寸 cm
        grid_scale: 网格周期 cm
        axis: 'x' (East-West) 或 'y' (North-South)
        bin_size: rate map bin大小 cm
        boundary_threshold: 边界接触阈值 cm
        n_bootstrap: 降采样重复次数
        threshold_ratio: 用于找峰的阈值比例
        
    返回:
        dict: 包含 shift_cm, shift_ratio, shift_std 等
    """
    if not np.isfinite(grid_scale) or grid_scale <= 0:
        return {'shift_cm': np.nan, 'shift_ratio': np.nan, 'error': 'Invalid grid scale'}

    # 识别边界接触
    boundary_labels = _identify_boundary_contacts(
        positions, env_size, boundary_threshold
    )
    
    # 创建边界数据
    n_bins_x = int(env_size[0] / bin_size)
    n_bins_y = int(env_size[1] / bin_size)
    
    boundary_data = _create_boundary_data(
        positions, spike_train, boundary_labels,
        n_bins_x, n_bins_y, bin_size
    )
    
    # 选择对立边界
    if axis == 'x':
        pair = ('East', 'West')
        corr_axis = 1
    else:
        pair = ('North', 'South')
        corr_axis = 0
    
    # 检查数据
    if boundary_data[pair[0]] is None or boundary_data[pair[1]] is None:
        return {'shift_cm': np.nan, 'shift_ratio': np.nan, 'error': 'Insufficient data'}
    
    # 最大滞后
    max_lag = int(grid_scale / (2 * bin_size))
    
    # Bootstrap计算
    shifts = []
    for i in range(n_bootstrap):
        map1, map2 = _compute_matched_rate_maps(
            boundary_data[pair[0]], boundary_data[pair[1]],
            n_bins_y, n_bins_x, seed=i
        )
        lags, corr = _cross_correlate_1d(map1, map2, corr_axis, max_lag)
        shift = _find_nearest_peak(lags, corr)
        # Fig3 convention: preserve sign per draw and apply abs after averaging.
        shifts.append(shift)
    
    mean_shift = np.nanmean(shifts)
    shift_cm = abs(mean_shift) * bin_size
    
    return {
        'shift_cm': shift_cm,
        'shift_ratio': shift_cm / grid_scale,
        'shift_std': np.nanstd(shifts) * bin_size,
        'n_bootstrap': n_bootstrap
    }


def compute_phase_shift_batch(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    grid_scale: Union[float, List[float], np.ndarray],
    axis: str = 'x',
    bin_size: float = 2.0,
    boundary_threshold: float = 12.0,
    n_bootstrap: int = 50,
    threshold_ratio: float=0.1
) -> np.ndarray:
    """
    批量计算多个细胞的相位差
    
    参数:
        cells_data: 列表，每个元素是 {'positions': array, 'spikes': array}
        env_size: 环境尺寸
        grid_scales: 单个值或与cells_data等长的列表
        axis: 'x' 或 'y'
        
    返回:
        shifts: (n_cells,) 相位差数组（相对于grid scale的比例）
    """
    n_cells = len(spike_train)
    
    
    scales = np.asarray(grid_scale, dtype=float)
    if scales.ndim == 0:
        scales = np.full(n_cells, float(scales))
    if scales.size != n_cells:
        raise ValueError("grid_scale must be scalar or contain one value per cell")
    shifts = np.zeros(n_cells)
    
    for i in range(n_cells):
        result = compute_phase_shift(
            positions=positions,
            spike_train=spike_train[i],
            env_size=env_size,
            grid_scale=float(scales[i]),
            axis=axis,
            bin_size=bin_size,
            boundary_threshold=boundary_threshold,
            n_bootstrap=n_bootstrap,
            threshold_ratio=threshold_ratio
        )
        shifts[i] = result['shift_ratio']
    
    return shifts


# ========== 内部辅助函数 ==========

def _identify_boundary_contacts(
    positions: np.ndarray,
    env_size: Tuple[float, float],
    threshold: float
) -> np.ndarray:
    """识别每个时间点最近接触的边界"""
    n_samples = len(positions)
    boundary_labels = np.zeros(n_samples, dtype=int)
    
    x, y = positions[:, 0], positions[:, 1]
    width, height = env_size
    
    current = 0
    for i in range(n_samples):
        if y[i] >= height - threshold:
            current = 1  # North
        elif y[i] <= threshold:
            current = 2  # South
        elif x[i] >= width - threshold:
            current = 3  # East
        elif x[i] <= threshold:
            current = 4  # West
        boundary_labels[i] = current
    
    return boundary_labels


def _create_boundary_data(
    positions: np.ndarray,
    spike_train: np.ndarray,
    boundary_labels: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    bin_size: float
) -> Dict:
    """为每个边界创建spike数据"""
    boundary_names = {1: 'North', 2: 'South', 3: 'East', 4: 'West'}
    boundary_data = {}
    
    for b_id, b_name in boundary_names.items():
        mask = boundary_labels == b_id
        
        if np.sum(mask) == 0:
            boundary_data[b_name] = None
            continue
        
        pos_subset = positions[mask]
        spikes_subset = spike_train[mask]
        
        bin_x = np.clip((pos_subset[:, 0] / bin_size).astype(int), 0, n_bins_x - 1)
        bin_y = np.clip((pos_subset[:, 1] / bin_size).astype(int), 0, n_bins_y - 1)
        
        spike_lists = [[[] for _ in range(n_bins_x)] for _ in range(n_bins_y)]
        for i in range(len(bin_x)):
            spike_lists[bin_y[i]][bin_x[i]].append(spikes_subset[i])
        
        boundary_data[b_name] = spike_lists
    
    return boundary_data


def _compute_matched_rate_maps(
    data1: list,
    data2: list,
    n_bins_y: int,
    n_bins_x: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """通过降采样创建采样匹配的rate maps"""
    if seed is not None:
        np.random.seed(seed)
    
    rate_map1 = np.full((n_bins_y, n_bins_x), np.nan)
    rate_map2 = np.full((n_bins_y, n_bins_x), np.nan)
    
    for yi in range(n_bins_y):
        for xi in range(n_bins_x):
            samples1 = data1[yi][xi]
            samples2 = data2[yi][xi]
            
            n1, n2 = len(samples1), len(samples2)
            if n1 == 0 or n2 == 0:
                continue
            
            n_match = min(n1, n2)
            idx1 = np.random.choice(n1, n_match, replace=False)
            idx2 = np.random.choice(n2, n_match, replace=False)
            
            rate_map1[yi, xi] = np.mean([samples1[i] for i in idx1])
            rate_map2[yi, xi] = np.mean([samples2[i] for i in idx2])
    
    rate_map1 = _nan_gaussian_filter(rate_map1, sigma=1.5)
    rate_map2 = _nan_gaussian_filter(rate_map2, sigma=1.5)
    return rate_map1, rate_map2


def _nan_gaussian_filter(rate_map: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(rate_map)
    values = np.where(valid, rate_map, 0.0)
    numerator = ndimage.gaussian_filter(values, sigma=sigma, mode='constant')
    denominator = ndimage.gaussian_filter(valid.astype(float), sigma=sigma, mode='constant')
    result = np.full_like(values, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    result[~valid] = np.nan
    return result


def _cross_correlate_1d(
    map1: np.ndarray,
    map2: np.ndarray,
    axis: int,
    max_lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """沿指定维度计算互相关"""
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if axis == 1:  # x方向
            if lag >= 0:
                s1 = map1[:, :map1.shape[1]-lag] if lag > 0 else map1
                s2 = map2[:, lag:] if lag > 0 else map2
            else:
                s1 = map1[:, -lag:]
                s2 = map2[:, :map2.shape[1]+lag]
        else:  # y方向
            if lag >= 0:
                s1 = map1[:map1.shape[0]-lag, :] if lag > 0 else map1
                s2 = map2[lag:, :] if lag > 0 else map2
            else:
                s1 = map1[-lag:, :]
                s2 = map2[:map2.shape[0]+lag, :]
        
        valid = ~(np.isnan(s1) | np.isnan(s2))
        if np.sum(valid) < 20:
            correlations[i] = np.nan
            continue
        
        v1, v2 = s1[valid], s2[valid]
        if np.std(v1) > 0 and np.std(v2) > 0:
            correlations[i] = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
        else:
            correlations[i] = np.nan
    
    return lags, correlations


def _find_nearest_peak(lags: np.ndarray, correlations: np.ndarray) -> float:
    """Return the strongest lag in the positive blob nearest zero (Fig3)."""
    valid_corr = correlations.copy()
    valid_corr[np.isnan(valid_corr)] = -1

    positive_mask = valid_corr > 0
    if not np.any(positive_mask):
        return 0.0

    labeled, n_blobs = label(positive_mask)
    if n_blobs == 0:
        return 0.0
    center_idx = len(lags) // 2
    best_blob = min(
        range(1, n_blobs + 1),
        key=lambda blob_id: abs(np.mean(np.where(labeled == blob_id)[0]) - center_idx),
    )
    blob_corr = valid_corr.copy()
    blob_corr[labeled != best_blob] = -np.inf
    return float(lags[np.argmax(blob_corr)])
