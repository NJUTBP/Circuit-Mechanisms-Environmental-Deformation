# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:37:23 2025

@author: 17534
"""
"""
网格细胞重缩放系数计算 - 单轴版本（仅x方向）

适用于环境只在一个维度（x方向）变形的情况：
- y方向固定为100cm
- x方向从70cm变化到130cm
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RescaleResult:
    """单个细胞的重缩放结果"""
    rescale_factor: float           # x方向重缩放系数
    best_correlation: float         # 最佳相关值
    best_scale_cm: float           # 最佳x方向缩放尺寸 (cm)
    best_lag_x: int                # 最佳x方向偏移 (bins)
    all_scales: np.ndarray         # 所有测试的缩放尺寸
    all_correlations: np.ndarray   # 对应的最大相关值


def compute_grid_rescale_factor(
    familiar_rate_map: np.ndarray,
    deformed_rate_map: np.ndarray,
    familiar_size_x: float,
    deformed_size_x: float,
    bin_size: float = 2.0,
    scale_range: float = 10.0,
    scale_step: float = 2.0,
    mode: str = 'valid',
    min_overlap_ratio: float = 0.5
) -> RescaleResult:
    """
    计算单个网格细胞在x方向的重缩放系数
    
    参数:
        familiar_rate_map: 熟悉环境的rate map (n_bins_y, n_bins_x)
        deformed_rate_map: 变形环境的rate map (n_bins_y, m_bins_x)
                          注意：y方向bins数相同，x方向不同
        familiar_size_x: 熟悉环境x方向尺寸 (cm)
        deformed_size_x: 变形环境x方向尺寸 (cm)
        bin_size: bin大小 (cm)
        scale_range: 缩放搜索范围，在min/max尺寸基础上±scale_range (cm)
        scale_step: 缩放步长 (cm)
        mode: 'valid' - x方向完全重叠，滑动取最大
              'partial' - x方向可部分超出，但重叠至少占50%
              'boundary' - 仅在左右边界对齐处计算，取二者最大值
        min_overlap_ratio: 'partial'模式下x方向最小重叠比例
        
    返回:
        RescaleResult: 包含重缩放系数等信息
    """
    # 计算x方向缩放尺寸范围
    min_size = min(familiar_size_x, deformed_size_x) - scale_range
    max_size = max(familiar_size_x, deformed_size_x) + scale_range
    min_size = max(min_size, 20)  # 最小20cm
    
    scale_sizes = np.arange(min_size, max_size + scale_step, scale_step)
    
    best_corr = -np.inf
    best_scale = familiar_size_x
    best_lag = 0
    all_max_corrs = []
    
    for scale_size in scale_sizes:
        # 仅在x方向缩放熟悉环境rate map
        scaled_map = _rescale_rate_map_x(
            familiar_rate_map, familiar_size_x, scale_size, bin_size
        )
        
        # 计算与变形环境的互相关（仅x方向滑动）
        if mode == 'valid':
            max_corr, lag_x = _compute_correlation_x_valid(
                scaled_map, deformed_rate_map
            )
        elif mode == 'partial':
            max_corr, lag_x = _compute_correlation_x_partial(
                scaled_map, deformed_rate_map, min_overlap_ratio
            )
        elif mode == 'boundary':
            max_corr, lag_x = _compute_correlation_x_boundary(
                scaled_map, deformed_rate_map
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'valid', 'partial', or 'boundary'")
        
        all_max_corrs.append(max_corr)
        
        if not np.isnan(max_corr) and max_corr > best_corr:
            best_corr = max_corr
            best_scale = scale_size
            best_lag = lag_x
    
    rescale_factor = best_scale / familiar_size_x
    
    return RescaleResult(
        rescale_factor=rescale_factor,
        best_correlation=best_corr,
        best_scale_cm=best_scale,
        best_lag_x=best_lag,
        all_scales=scale_sizes,
        all_correlations=np.array(all_max_corrs)
    )


def compute_grid_rescale_factor_batch(
    familiar_rate_maps: List[np.ndarray],
    deformed_rate_maps: List[np.ndarray],
    familiar_size_x: float,
    deformed_size_x: float,
    bin_size: float = 2.0,
    scale_range: float = 10.0,
    scale_step: float = 2.0,
    mode: str = 'valid',
    min_overlap_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算一组网格细胞的x方向重缩放系数
    
    返回:
        rescale_factors: (n_cells,) 重缩放系数数组
        best_correlations: (n_cells,) 最佳相关值数组
    """
    n_cells = len(familiar_rate_maps)
    rescale_factors = np.zeros(n_cells)
    best_correlations = np.zeros(n_cells)
    
    for i in range(n_cells):
        result = compute_grid_rescale_factor(
            familiar_rate_maps[i],
            deformed_rate_maps[i],
            familiar_size_x,
            deformed_size_x,
            bin_size,
            scale_range,
            scale_step,
            mode,
            min_overlap_ratio
        )
        rescale_factors[i] = result.rescale_factor
        best_correlations[i] = result.best_correlation
    
    return rescale_factors, best_correlations


def compute_all_rescale_factors(
    module_data: Dict[int, List[np.ndarray]],
    familiar_size_x: float = 100,
    env_sizes_x: List[int] = None,
    bin_size: float = 2.0,
    scale_range: float = 10.0,
    scale_step: float = 2.0,
    mode: str = 'valid',
    min_overlap_ratio: float = 0.5,
    verbose: bool = True
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    计算一个模块在所有环境的x方向重缩放系数
    
    参数:
        module_data: {
            70: [rate_map1, rate_map2, ...],   # n个细胞，x方向70cm
            80: [...],
            100: [...],  # 熟悉环境，x方向100cm
            ...
        }
        familiar_size_x: 熟悉环境x方向尺寸 (cm)
        env_sizes_x: 要计算的环境x方向尺寸列表
        
    返回:
        results: {
            70: (rescale_factors, correlations),
            80: (...),
            ...
        }
    """
    if env_sizes_x is None:
        env_sizes_x = [s for s in module_data.keys() if s != familiar_size_x]
    
    familiar_maps = module_data[familiar_size_x]
    results = {}
    
    for i, env_size_x in enumerate(env_sizes_x):
        if verbose:
            print(f"\r[{i+1}/{len(env_sizes_x)}] {env_size_x}cm...", end='')
        
        deformed_maps = module_data[env_size_x]
        
        factors, corrs = compute_grid_rescale_factor_batch(
            familiar_maps,
            deformed_maps,
            familiar_size_x,
            env_size_x,
            bin_size,
            scale_range,
            scale_step,
            mode,
            min_overlap_ratio
        )
        
        results[env_size_x] = (factors, corrs)
    
    if verbose:
        print(" Done!")
    
    return results


# ========== 内部辅助函数 ==========

def _rescale_rate_map_x(
    rate_map: np.ndarray,
    original_size_x: float,
    target_size_x: float,
    bin_size: float
) -> np.ndarray:
    """
    仅在x方向缩放rate map，y方向保持不变
    """
    # 处理NaN
    rate_map_filled = rate_map.copy()
    nan_mask = np.isnan(rate_map_filled)
    rate_map_filled[nan_mask] = 0
    
    # 原始和目标x方向bins数
    orig_bins_x = rate_map.shape[1]
    target_bins_x = int(target_size_x / bin_size)
    
    # 缩放因子：仅x方向
    scale_x = target_bins_x / orig_bins_x
    
    if scale_x == 1.0:
        return rate_map.copy()
    
    # 使用zoom仅缩放x方向 (y方向缩放因子=1)
    # Supplementary Note 4 specifies bilinear interpolation.
    scaled_map = zoom(rate_map_filled, (1.0, scale_x), order=1)
    
    # 处理NaN掩码
    if nan_mask.any():
        nan_mask_scaled = zoom(nan_mask.astype(float), (1.0, scale_x), order=0) > 0.5
        scaled_map[nan_mask_scaled] = np.nan
    
    return scaled_map


def _compute_correlation_x_valid(
    map1: np.ndarray,
    map2: np.ndarray
) -> Tuple[float, int]:
    """
    计算两个rate map的最大相关值（valid模式）
    仅在x方向滑动，y方向完全对齐
    """
    h1, w1 = map1.shape
    h2, w2 = map2.shape
    
    # y方向必须相同
    if h1 != h2:
        min_h = min(h1, h2)
        map1 = map1[:min_h, :]
        map2 = map2[:min_h, :]
    
    w1, w2 = map1.shape[1], map2.shape[1]
    
    # 确定哪个更宽
    if w1 <= w2:
        small_map, large_map = map1, map2
        ws, wl = w1, w2
        swapped = False
    else:
        small_map, large_map = map2, map1
        ws, wl = w2, w1
        swapped = True
    
    # x方向滑动范围
    max_lag_x = wl - ws
    
    if max_lag_x < 0:
        return _compute_pearson(map1, map2), 0
    
    best_corr = -np.inf
    best_lag = 0
    
    for lag_x in range(max_lag_x + 1):
        # 从宽map中提取与窄map对应的区域
        region = large_map[:, lag_x:lag_x+ws]
        corr = _compute_pearson(small_map, region)
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = -lag_x if swapped else lag_x
    
    return best_corr, best_lag


def _compute_correlation_x_partial(
    map1: np.ndarray,
    map2: np.ndarray,
    min_overlap_ratio: float = 0.5
) -> Tuple[float, int]:
    """
    计算两个rate map的最大相关值（partial模式）
    仅在x方向滑动，允许部分超出
    """
    h1, w1 = map1.shape
    h2, w2 = map2.shape
    
    # y方向必须相同
    if h1 != h2:
        min_h = min(h1, h2)
        map1 = map1[:min_h, :]
        map2 = map2[:min_h, :]
    
    w1, w2 = map1.shape[1], map2.shape[1]
    
    # 最小重叠宽度
    min_overlap_w = int(max(w1, w2) * min_overlap_ratio)
    
    best_corr = -np.inf
    best_lag = 0
    
    # x方向滑动：lag_x是map2相对于map1的偏移
    for lag_x in range(-w2 + min_overlap_w, w1 - min_overlap_w + 1):
        # 计算重叠区域
        x1_start = max(0, lag_x)
        x1_end = min(w1, w2 + lag_x)
        x2_start = max(0, -lag_x)
        x2_end = x2_start + (x1_end - x1_start)
        
        overlap_w = x1_end - x1_start
        
        if overlap_w < min_overlap_w:
            continue
        
        # 检查重叠比例
        if overlap_w < w1 * min_overlap_ratio or overlap_w < w2 * min_overlap_ratio:
            continue
        
        region1 = map1[:, x1_start:x1_end]
        region2 = map2[:, x2_start:x2_end]
        
        corr = _compute_pearson(region1, region2)
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag_x
    
    return best_corr, best_lag


def _compute_correlation_x_boundary(
    map1: np.ndarray,
    map2: np.ndarray
) -> Tuple[float, int]:
    """
    计算两个rate map在左右边界对齐时的相关值（boundary模式）
    
    按照文献方法：
    1. 左边界对齐：两个map的左边缘对齐
    2. 右边界对齐：两个map的右边缘对齐
    3. 取二者中相关值较大者
    
    返回:
        max_corr: 两种对齐方式中的最大相关值
        best_alignment: 0=左边界对齐, 1=右边界对齐（用lag_x表示）
    """
    h1, w1 = map1.shape
    h2, w2 = map2.shape
    
    # y方向必须相同
    if h1 != h2:
        min_h = min(h1, h2)
        map1 = map1[:min_h, :]
        map2 = map2[:min_h, :]
    
    w1, w2 = map1.shape[1], map2.shape[1]
    
    # 重叠宽度（取较小者）
    overlap_w = min(w1, w2)
    
    # 左边界对齐：两个map都从左边开始取overlap_w宽度
    region1_left = map1[:, :overlap_w]
    region2_left = map2[:, :overlap_w]
    corr_left = _compute_pearson(region1_left, region2_left)
    
    # 右边界对齐：两个map都从右边开始取overlap_w宽度
    region1_right = map1[:, -overlap_w:]
    region2_right = map2[:, -overlap_w:]
    corr_right = _compute_pearson(region1_right, region2_right)
    
    # 取最大值
    if np.isnan(corr_left) and np.isnan(corr_right):
        return np.nan, 0
    elif np.isnan(corr_left):
        return corr_right, 1  # 右边界对齐
    elif np.isnan(corr_right):
        return corr_left, 0   # 左边界对齐
    elif corr_left >= corr_right:
        return corr_left, 0   # 左边界对齐
    else:
        return corr_right, 1  # 右边界对齐


def _compute_pearson(map1: np.ndarray, map2: np.ndarray) -> float:
    """计算两个rate map的Pearson相关系数"""
    # 处理NaN
    valid = ~(np.isnan(map1) | np.isnan(map2))
    
    if np.sum(valid) < 20:
        return np.nan
    
    v1 = map1[valid].flatten()
    v2 = map2[valid].flatten()
    
    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    
    return np.corrcoef(v1, v2)[0, 1]


# ========== 可视化 ==========
def visualize_rescale_analysis(
    familiar_map: np.ndarray,
    deformed_map: np.ndarray,
    result: RescaleResult,
    familiar_size_x: float,
    deformed_size_x: float,
    size_y: float = 100,
    bin_size: float = 2.0,
    save_path: Optional[str] = None
):

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    # === 关键修改开始 ===
    width_ratios = [
        familiar_size_x,
        deformed_size_x,
        result.best_scale_cm
    ]

    fig = plt.figure(figsize=(12,7))
    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1, 1],
        wspace=0.25,
        hspace=0.35,
    )

    axes = np.empty((2,3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i,j] = fig.add_subplot(gs[i,j])

    '''
def visualize_rescale_analysis(
    familiar_map: np.ndarray,
    deformed_map: np.ndarray,
    result: RescaleResult,
    familiar_size_x: float,
    deformed_size_x: float,
    size_y: float = 100,
    bin_size: float = 2.0,
    save_path: Optional[str] = None
):
    """可视化x方向重缩放分析结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    '''
    
    # 1. 熟悉环境rate map
    ax = axes[0, 0]
    im = ax.imshow(familiar_map, origin='lower', cmap='jet',
                  extent=[0, familiar_size_x, 0, size_y], aspect='auto')
    ax.set_title(f'Familiar ({familiar_size_x}×{size_y}cm)')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    plt.colorbar(im, ax=ax)
    
    # 2. 变形环境rate map
    ax = axes[0, 1]
    im = ax.imshow(deformed_map, origin='lower', cmap='jet',
                  extent=[0, deformed_size_x, 0, size_y], aspect='auto')
    ax.set_title(f'Deformed ({deformed_size_x}×{size_y}cm)')
    ax.set_xlabel('X (cm)')
    #ax.set_ylabel('Y (cm)')
    plt.colorbar(im, ax=ax)
    
    # 3. 最佳缩放后的熟悉环境rate map
    ax = axes[0, 2]
    best_scaled_map = _rescale_rate_map_x(
        familiar_map, familiar_size_x, result.best_scale_cm, bin_size
    )
    im = ax.imshow(best_scaled_map, origin='lower', cmap='jet',
                  extent=[0, result.best_scale_cm, 0, size_y], aspect='auto')
    ax.set_title(f'Rescaled Familiar ({result.best_scale_cm:.0f}×{size_y}cm)')
    ax.set_xlabel('X (cm)')
    #ax.set_ylabel('Y (cm)')
    plt.colorbar(im, ax=ax, label='Hz')
    
    # 4. 缩放比例 vs 相关值曲线
    ax = axes[1, 0]
    ax.plot(result.all_scales, result.all_correlations, 'b-', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=result.best_scale_cm, color='r', linestyle='--', linewidth=1.5,
              label=f'Best: {result.best_scale_cm:.0f}cm')
    ax.axvline(x=familiar_size_x, color='g', linestyle=':', alpha=0.7,
              label=f'Familiar: {familiar_size_x:.0f}cm')
    ax.axvline(x=deformed_size_x, color='orange', linestyle=':', alpha=0.7,
              label=f'Deformed: {deformed_size_x:.0f}cm')
    ax.set_xlabel('Rescaled X Size (cm)')
    ax.set_ylabel('Max Correlation')
    ax.set_title('Correlation vs X-Rescale')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. 结果信息
    ax = axes[1, 1]
    ax.axis('off')
    
    # 判断对齐方式（仅boundary模式有意义）
    alignment_text = ""
    if hasattr(result, 'best_lag_x'):
        if result.best_lag_x == 0:
            alignment_text = "\nBest Alignment: Left boundary"
        elif result.best_lag_x == 1:
            alignment_text = "\nBest Alignment: Right boundary"
        else:
            alignment_text = f"\nBest X Offset: {result.best_lag_x} bins"
    
    info_text = (
        f"X-Rescale Factor: {result.rescale_factor:.3f}\n\n"
        f"Best Correlation: {result.best_correlation:.3f}\n\n"
        #f"Best X Scale: {result.best_scale_cm:.1f} cm"
        f"{alignment_text}\n\n"
        f"Expected (geometric): {deformed_size_x/familiar_size_x:.3f}"
    )
    ax.text(0, 0.5, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax.set_title('Analysis Results')
    
    # 6. 预期vs实际对比
    ax = axes[1, 2]
    expected = deformed_size_x / familiar_size_x
    actual = result.rescale_factor
    
    bars = ax.bar(['Expected\n(Geometric)', 'Actual\n(Measured)'], 
                  [expected, actual],
                  color=['steelblue', 'coral'])
    #ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Rescale Factor')
    ax.set_ylim([0,1])
    ax.set_title('Expected vs Actual')
    
    for bar, val in zip(bars, [expected, actual]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

# ========== 演示 ==========

def demo():
    """演示重缩放系数计算"""
    import matplotlib.pyplot as plt
    import ratemap_cal
    print("=" * 60)
    print("网格细胞重缩放系数计算演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 参数
    familiar_size = 100  # cm
    deformed_size = 80   # cm (压缩)
    bin_size = 2.0
    
    spiking_gc_10=np.load(r'D:\References\codings\Remapping\rescale_10\%s10cm\init\spikingrgc%d_init_%s10cm_norm30363_greo_e051204_9.npy'%(10,3,10))
    trace_10=np.load(r'D:\References\codings\Remapping\rescale_10\traj_%s10cm_dt10_1000s_1_1.npy'%(10),allow_pickle=True).item()
    Pos_10=trace_10['pos'][::10]
    
    spiking_gc_80=np.load(r'D:\References\codings\Remapping\rescale_10\%s10cm\init\spikingrgc%d_init_%s10cm_norm30363_greo_e051204_9.npy'%(80,3,80))
    trace_80=np.load(r'D:\References\codings\Remapping\rescale_10\traj_%s10cm_dt10_1000s_1_1.npy'%(80),allow_pickle=True).item()
    Pos_80=trace_80['pos'][::10]
    
    rate_map_10=ratemap_cal.cal_ratemap(spiking_gc_10,Pos_10,[0,1,0,1],0.02)
    rate_map_80=ratemap_cal.cal_ratemap(spiking_gc_80,Pos_80,[0,0.8,0,1],0.02)
    
    
    '''
    spikingrategc1_10=np.load(r'D:\References\codings\Remapping\rescale_10\init_1126\1010init_bp28\1010cm\spikingrgc3_init_1010cm_norm31261_greo_al15_Th1_e5_051202_bp_1_5.npy')
    trace_10=np.load(r'D:\References\codings\Remapping\rescale_10\traj_1010cm_dt10_1000s_2_%d.npy'%(1),allow_pickle=True).item()    
    Pos_10=trace_10['pos'][::10]
    spikingrategc1_80=np.load(r'D:\References\codings\Remapping\rescale_10\init_1126\1010init_bp28\8010cm\spikingrgc3_init_8010cm_norm31261_greo_al15_Th1_e5_051202_bp_1_5.npy')
    trace_80=np.load(r'D:\References\codings\Remapping\rescale_10\traj_8010cm_dt10_1000s_1_%d.npy'%(1),allow_pickle=True).item()    
    Pos_80=trace_80['pos'][::10]

    rate_map_10=ratemap_cal.cal_ratemap(spikingrategc1_10,Pos_10,[0,1,0,1],0.02)
    rate_map_80=ratemap_cal.cal_ratemap(spikingrategc1_80,Pos_80,[0,0.8,0,1],0.02)
    '''
    
    '''
    spikingrategc1_10=np.load(r'D:\References\codings\Remapping\rescale_10\init_1126\1010cm\spikingrgc1_init_1010cm_norm30363_greo_al15_Th1_e5_051202_bp_2_5.npy')
    trace_10=np.load(r'D:\References\codings\Remapping\rescale_10\traj_1010cm_dt10_1000s_2_%d.npy'%(1),allow_pickle=True).item()    
    Pos_10=trace_10['pos'][::10]
    spikingrategc1_80=np.load(r'D:\References\codings\Remapping\rescale_10\init_1126\8010cm\spikingrgc1_fam_8010cm_norm30363_greo_al15_Th1_e5_051202_bp_2_5.npy')
    trace_80=np.load(r'D:\References\codings\Remapping\rescale_10\traj_8010cm_dt10_1000s_1_%d.npy'%(1),allow_pickle=True).item()    
    Pos_80=trace_80['pos'][::10]

    rate_map_10=ratemap_cal.cal_ratemap(spikingrategc1_10,Pos_10,[0,1,0,1],0.02)
    rate_map_80=ratemap_cal.cal_ratemap(spikingrategc1_80,Pos_80,[0,0.8,0,1],0.02)
    '''
    '''
    spiking_gc_10=np.load(r'D:\References\codings\Remapping\rescale_10\%s10cm\init\spikingrgc%d_init_%s10cm_norm30363_greo_e051204_9.npy'%(10,3,10))
    trace_10=np.load(r'D:\References\codings\Remapping\rescale_10\traj_%s10cm_dt10_1000s_1_1.npy'%(10),allow_pickle=True).item()
    Pos_10=trace_10['pos'][::10]
    spiking_gc_80=np.load(r'D:\References\codings\Remapping\rescale_10\8010cm\fam\spikingr\spikingrgc%d_fam_8010cm_norm30363_greo_e051204_%d.npy'%(3,9))
    trace_80=np.load(r'D:\References\codings\Remapping\rescale_10\traj_8010cm_dt10_1000s_1_1.npy',allow_pickle=True).item()
    Pos_80=trace_80['pos'][::10]

    rate_map_10=ratemap_cal.cal_ratemap(spiking_gc_10,Pos_10,[0,1,0,1],0.02)
    rate_map_80=ratemap_cal.cal_ratemap(spiking_gc_80,Pos_80,[0,0.8,0,1],0.02)
    '''
    
    familiar_map=rate_map_10[1]
    deformed_map=rate_map_80[1]
    
    result_valid = compute_grid_rescale_factor(
        familiar_map, deformed_map,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='valid'
    )
    
    result_valid_batch = compute_grid_rescale_factor_batch(
        rate_map_10,
        rate_map_80,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='valid',
        min_overlap_ratio= 0.5
    )
    
    print(f"  result_valid_rescale: {result_valid_batch[0].mean():.3f}")
    print(f"  result_valid_rescale_std: {result_valid_batch[0].std():.3f}")
    print(f"  result_valid_corr: {result_valid_batch[1].mean():.3f}")
    
    
    print(f"  检测到的重缩放系数: {result_valid.rescale_factor:.3f}")
    print(f"  最佳相关值: {result_valid.best_correlation:.3f}")
    print(f"  最佳缩放尺寸: {result_valid.best_scale_cm:.1f} cm")
    
    # 可视化
    visualize_rescale_analysis(
        familiar_map, deformed_map, result_valid,
        familiar_size, deformed_size,
        bin_size=bin_size,
        #scale_range=15.0,
        #scale_step=2.0,
        #save_path='/mnt/user-data/outputs/rescale_analysis_valid.png'
    )
    
    # ========== 测试 partial 模式 ==========
    print("\n" + "-" * 40)
    print("Partial 模式 (允许部分超出)")
    print("-" * 40)
    
    result_partial = compute_grid_rescale_factor(
        familiar_map, deformed_map,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='partial',
        min_overlap_ratio=0.5
    )
    
    result_partial_batch = compute_grid_rescale_factor_batch(
        rate_map_10,
        rate_map_80,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='partial',
        min_overlap_ratio= 0.5
    )
    print(f"  result_partial_rescale: {result_partial_batch[0].mean():.3f}")
    print(f"  result_partial_corr: {result_partial_batch[1].mean():.3f}")
    
    
    print(f"  检测到的重缩放系数: {result_partial.rescale_factor:.3f}")
    print(f"  最佳相关值: {result_partial.best_correlation:.3f}")
    print(f"  最佳缩放尺寸: {result_partial.best_scale_cm:.1f} cm")
    
    visualize_rescale_analysis(
        familiar_map, deformed_map, result_partial,
        familiar_size, deformed_size,
        bin_size=bin_size,
        #scale_range=15.0,
        #scale_step=2.0,
        #save_path='/mnt/user-data/outputs/rescale_analysis_partial.png'
    )
    
    result_boundary = compute_grid_rescale_factor(
        familiar_map, deformed_map,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='boundary',
        min_overlap_ratio=0.5
    )
    
    result_boundary_batch = compute_grid_rescale_factor_batch(
        rate_map_10,
        rate_map_80,
        familiar_size, deformed_size,
        bin_size=bin_size,
        scale_range=15.0,
        scale_step=2.0,
        mode='boundary',
        min_overlap_ratio= 0.5
    )
    print(f"  result_boundary_rescale: {result_boundary_batch[0].mean():.3f}")
    print(f"  result_boundary_corr: {result_boundary_batch[1].mean():.3f}")
    
    
    print(f"  检测到的重缩放系数: {result_boundary.rescale_factor:.3f}")
    print(f"  最佳相关值: {result_boundary.best_correlation:.3f}")
    print(f"  最佳缩放尺寸: {result_boundary.best_scale_cm:.1f} cm")
    visualize_rescale_analysis(
        familiar_map, deformed_map, result_boundary,
        familiar_size, deformed_size,
        bin_size=bin_size,
        #scale_range=15.0,
        #scale_step=2.0,
        #save_path='/mnt/user-data/outputs/rescale_analysis_partial.png'
    )


if __name__ == "__main__":
    demo()
