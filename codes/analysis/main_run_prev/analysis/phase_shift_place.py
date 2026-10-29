# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:02:13 2025

@author: 17534
"""

"""
位置细胞相位差计算 - 基于速度方向分类

与网格细胞的边界接触分类不同，位置细胞使用速度方向分类：
- 当速度方向在指定角度范围内且速度大小超过阈值时，将该时间点分类到对应方向
- 比较对立方向（左vs右，上vs下）的放电野相位差
"""

import numpy as np
from scipy.ndimage import label, gaussian_filter
from scipy.signal import correlate2d
from typing import Tuple, Dict, Optional, Union, List
from scipy import ndimage

# ========== 核心计算函数 ==========

def compute_place_cell_phase_shift(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    axis: str = 'x',
    bin_size: float = 2.0,
    angle_range: float = 45.0,
    min_speed: float = 5.0,
    n_bootstrap: int = 100,
    dt: float = 0.02
) -> Dict:
    """
    计算位置细胞在指定轴向的相位差（基于速度方向分类）
    
    参数:
        positions: (n_samples, 2) 动物位置 [x, y]，单位cm
        spike_train: (n_samples,) 每个时间bin的spike数
        env_size: (width, height) 环境尺寸，单位cm
        axis: 'x' (左右方向) 或 'y' (上下方向)
        bin_size: rate map bin大小，单位cm
        angle_range: 速度方向的角度容差，单位度
                     例如45度表示速度方向在目标方向±45度内
        min_speed: 最小速度阈值，单位cm/s
        n_bootstrap: 降采样重复次数
        dt: 时间步长，单位秒（用于计算速度）
        
    返回:
        dict: 包含 shift_cm, shift_pixels, direction_counts 等
    """
    # 计算速度
    velocities = _compute_velocity(positions, dt)
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    
    # 按速度方向分类
    direction_labels = _classify_by_velocity(
        velocities, speeds, axis, angle_range, min_speed
    )
    
    # 创建方向数据
    n_bins_x = int(env_size[0] / bin_size)
    n_bins_y = int(env_size[1] / bin_size)
    
    direction_data = _create_direction_data(
        positions, spike_train, direction_labels,
        n_bins_x, n_bins_y, bin_size
    )
    
    # 选择对立方向
    if axis == 'x':
        pair = ('right', 'left')
        corr_axis = 1
    else:
        pair = ('up', 'down')
        corr_axis = 0
    
    # 检查数据充足性
    if direction_data[pair[0]] is None or direction_data[pair[1]] is None:
        return {
            'shift_cm': np.nan,
            'shift_pixels': np.nan,
            'error': 'Insufficient data for one or both directions'
        }
    
    # 统计各方向样本数
    direction_counts = {
        'right': np.sum(direction_labels == 1),
        'left': np.sum(direction_labels == 2),
        'up': np.sum(direction_labels == 3),
        'down': np.sum(direction_labels == 4),
        'unclassified': np.sum(direction_labels == 0)
    }
    
    # Bootstrap计算相位差
    # 使用rate map的互相关（不限制搜索范围，因为位置细胞没有固定周期）
    max_lag = min(n_bins_x, n_bins_y) // 2  # 最大搜索半个环境
    
    shifts = []
    for i in range(n_bootstrap):
        map1, map2 = _compute_matched_rate_maps(
            direction_data[pair[0]], direction_data[pair[1]],
            n_bins_y, n_bins_x, seed=i
        )
        
        # 计算互相关
        lags, corr = _cross_correlate_1d(map1, map2, corr_axis, max_lag)
        
        # 找到峰值位置
        shift = _find_correlation_peak(lags, corr)
        shifts.append(shift)
    
    mean_shift = np.nanmean(shifts)
    shift_cm = mean_shift * bin_size
    
    return {
        'shift_pixels': mean_shift,
        'shift_cm': shift_cm,
        'shift_std_cm': np.nanstd(shifts) * bin_size,
        'direction_pair': pair,
        'direction_counts': direction_counts,
        'n_bootstrap': n_bootstrap,
        'params': {
            'angle_range': angle_range,
            'min_speed': min_speed,
            'bin_size': bin_size
        }
    }


def compute_place_cell_phase_shift_batch(
    positions: np.ndarray,
    spike_train: np.ndarray,
    env_size: Tuple[float, float],
    axis: str = 'x',
    bin_size: float = 2.0,
    angle_range: float = 45.0,
    min_speed: float = 5.0,
    n_bootstrap: int = 100,
    dt: float = 0.02
) -> np.ndarray:
    """
    批量计算多个位置细胞的相位差
    
    参数:
        cells_data: 列表，每个元素是 {'positions': array, 'spikes': array}
        其他参数同 compute_place_cell_phase_shift
        
    返回:
        shifts: (n_cells,) 相位差数组（单位cm）
    """
    n_cells = len(spike_train)
    shifts = np.zeros(n_cells)
    
    for i in range(n_cells):
        result = compute_place_cell_phase_shift(
            positions=positions,
            spike_train=spike_train[i],
            env_size=env_size,
            axis=axis,
            bin_size=bin_size,
            angle_range=angle_range,
            min_speed=min_speed,
            n_bootstrap=n_bootstrap,
            dt=dt
        )
        shifts[i] = result['shift_cm']
    
    return shifts


def compute_all_place_cell_phase_shifts(
    env_data: Dict[int, List[Dict]],
    env_sizes: List[int],
    axes: List[str] = ['x', 'y'],
    bin_size: float = 2.0,
    angle_range: float = 45.0,
    min_speed: float = 5.0,
    n_bootstrap: int = 100,
    dt: float = 0.02,
    verbose: bool = True
) -> Dict:
    """
    一次性计算所有环境、所有轴向的位置细胞相位差
    
    参数:
        env_data: {
            70: [{'positions': array, 'spikes': array}, ...],  # n个细胞
            80: [...],
            ...
        }
        env_sizes: [70, 80, 90, 100, 110, 120, 130]
        axes: ['x', 'y']
        
    返回:
        results: {
            'x': {70: array(n,), 80: array(n,), ...},
            'y': {70: array(n,), 80: array(n,), ...}
        }
    """
    results = {ax: {} for ax in axes}
    
    total = len(env_sizes) * len(axes)
    count = 0
    
    for env_size_val in env_sizes:
        env_size = (env_size_val, env_size_val)
        cells = env_data[env_size_val]
        
        for axis in axes:
            count += 1
            if verbose:
                print(f"\r[{count}/{total}] {env_size_val}cm, {axis}-axis", end='')
            
            shifts = compute_place_cell_phase_shift_batch(
                cells_data=cells,
                env_size=env_size,
                axis=axis,
                bin_size=bin_size,
                angle_range=angle_range,
                min_speed=min_speed,
                n_bootstrap=n_bootstrap,
                dt=dt
            )
            
            results[axis][env_size_val] = shifts
    
    if verbose:
        print("\nDone!")
    
    return results


# ========== 内部辅助函数 ==========

def _compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    计算每个时间点的速度向量
    
    使用中心差分：v[i] = (pos[i+1] - pos[i-1]) / (2*dt)
    边界处使用前向/后向差分
    """
    n_samples = len(positions)
    velocities = np.zeros((n_samples, 2))
    
    # 中心差分
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    
    # 边界处理
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    return velocities


def _classify_by_velocity(
    velocities: np.ndarray,
    speeds: np.ndarray,
    axis: str,
    angle_range: float,
    min_speed: float
) -> np.ndarray:
    """
    按速度方向分类轨迹点
    
    返回:
        labels: 0=未分类, 1=right, 2=left, 3=up, 4=down
    """
    n_samples = len(velocities)
    labels = np.zeros(n_samples, dtype=int)
    
    # 计算速度方向角度（度）
    # 0度=右，90度=上，180/-180=左，-90度=下
    angles = np.arctan2(velocities[:, 1], velocities[:, 0]) * 180 / np.pi
    
    # 速度足够大的点才分类
    fast_enough = speeds >= min_speed
    
    # 定义各方向的中心角度
    direction_centers = {
        'right': 0,
        'left': 180,  # 或-180
        'up': 90,
        'down': -90
    }
    
    # 角度范围转换为弧度比较更准确，但这里用度数更直观
    angle_tol = angle_range
    
    for i in range(n_samples):
        if not fast_enough[i]:
            continue
        
        ang = angles[i]
        
        # 检查右方向
        if abs(ang - 0) <= angle_tol:
            labels[i] = 1  # right
        # 检查左方向（需要处理±180的跳变）
        elif abs(ang - 180) <= angle_tol or abs(ang + 180) <= angle_tol:
            labels[i] = 2  # left
        # 检查上方向
        elif abs(ang - 90) <= angle_tol:
            labels[i] = 3  # up
        # 检查下方向
        elif abs(ang + 90) <= angle_tol:
            labels[i] = 4  # down
    
    return labels


def _create_direction_data(
    positions: np.ndarray,
    spike_train: np.ndarray,
    direction_labels: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    bin_size: float
) -> Dict:
    """
    为每个方向创建spike数据
    """
    direction_names = {1: 'right', 2: 'left', 3: 'up', 4: 'down'}
    direction_data = {}
    
    for d_id, d_name in direction_names.items():
        mask = direction_labels == d_id
        
        if np.sum(mask) < 100:  # 至少需要100个样本
            direction_data[d_name] = None
            continue
        
        pos_subset = positions[mask]
        spikes_subset = spike_train[mask]
        
        bin_x = np.clip((pos_subset[:, 0] / bin_size).astype(int), 0, n_bins_x - 1)
        bin_y = np.clip((pos_subset[:, 1] / bin_size).astype(int), 0, n_bins_y - 1)
        
        spike_lists = [[[] for _ in range(n_bins_x)] for _ in range(n_bins_y)]
        for i in range(len(bin_x)):
            spike_lists[bin_y[i]][bin_x[i]].append(spikes_subset[i])
        
        direction_data[d_name] = spike_lists
    
    return direction_data


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
    
    rate_map1 = np.full((n_bins_y, n_bins_x), 0)
    rate_map2 = np.full((n_bins_y, n_bins_x), 0)
    
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
    
    #rate_map1 = ndimage.gaussian_filter(rate_map1, sigma=1.5)
    #rate_map2 = ndimage.gaussian_filter(rate_map2, sigma=1.5)
    return rate_map1, rate_map2


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


def _find_correlation_peak(lags: np.ndarray, correlations: np.ndarray) -> float:
    """
    找到互相关中距离中心最近的峰值
    
    与网格细胞方法相同，使用连通区域分析
    """
    valid_corr = correlations.copy()
    valid_corr[np.isnan(valid_corr)] = -1
    
    if np.all(valid_corr == -1):
        return 0.0
    return float(lags[np.argmax(valid_corr)])


# ========== 可视化函数 ==========

def visualize_velocity_classification(
    positions: np.ndarray,
    velocities: np.ndarray,
    direction_labels: np.ndarray,
    env_size: Tuple[float, float],
    save_path: Optional[str] = None
):
    """
    可视化速度方向分类结果
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 颜色映射
    colors = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    labels_text = {0: 'Unclassified', 1: 'Right', 2: 'Left', 3: 'Up', 4: 'Down'}
    
    # 1. 轨迹按方向着色
    ax = axes[0]
    for d_id in range(5):
        mask = direction_labels == d_id
        if np.sum(mask) > 0:
            ax.scatter(positions[mask, 0], positions[mask, 1],
                      c=colors[d_id], s=1, alpha=0.3, label=labels_text[d_id])
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Trajectory by Velocity Direction')
    ax.legend(markerscale=5, loc='upper right')
    ax.set_aspect('equal')
    
    # 2. 速度方向分布（极坐标）
    ax = axes[1]
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    
    # 只显示分类的点
    classified = direction_labels > 0
    ax.hist(angles[classified] * 180 / np.pi, bins=72, 
           color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='Right (0°)')
    ax.axvline(x=180, color='blue', linestyle='--', label='Left (±180°)')
    ax.axvline(x=-180, color='blue', linestyle='--')
    ax.axvline(x=90, color='green', linestyle='--', label='Up (90°)')
    ax.axvline(x=-90, color='orange', linestyle='--', label='Down (-90°)')
    ax.set_xlabel('Velocity Direction (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Direction Distribution')
    ax.legend(fontsize=7)
    
    # 3. 各方向统计
    ax = axes[2]
    counts = [np.sum(direction_labels == i) for i in range(5)]
    bars = ax.bar(range(5), counts, color=[colors[i] for i in range(5)])
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Unclass.', 'Right', 'Left', 'Up', 'Down'])
    ax.set_ylabel('Sample Count')
    ax.set_title('Direction Classification Summary')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        #plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_direction_rate_maps(
    positions: np.ndarray,
    spike_train: np.ndarray,
    direction_labels: np.ndarray,
    env_size: Tuple[float, float],
    bin_size: float = 2.0,
    save_path: Optional[str] = None
):
    """
    可视化各方向的rate map
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    n_bins_x = int(env_size[0] / bin_size)
    n_bins_y = int(env_size[1] / bin_size)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    direction_names = {0: 'All', 1: 'Right', 2: 'Left', 3: 'Up', 4: 'Down'}
    subplot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    all_vmax = 0
    rate_maps = {}
    
    # 计算所有rate maps
    for d_id, d_name in direction_names.items():
        if d_id == 0:
            mask = np.ones(len(positions), dtype=bool)
        else:
            mask = direction_labels == d_id
        
        if np.sum(mask) < 10:
            rate_maps[d_name] = None
            continue
        
        pos_sub = positions[mask]
        spk_sub = spike_train[mask]
        
        bin_x = np.clip((pos_sub[:, 0] / bin_size).astype(int), 0, n_bins_x - 1)
        bin_y = np.clip((pos_sub[:, 1] / bin_size).astype(int), 0, n_bins_y - 1)
        
        spike_map = np.zeros((n_bins_y, n_bins_x))
        occ_map = np.zeros((n_bins_y, n_bins_x))
        
        for i in range(len(bin_x)):
            spike_map[bin_y[i], bin_x[i]] += spk_sub[i]
            occ_map[bin_y[i], bin_x[i]] += 1
        
        # 平滑
        #spike_map = gaussian_filter(spike_map, 1)
        #occ_map = gaussian_filter(occ_map, 1)
        occ_map[occ_map < 0.01] = 0
        
        rate_map = np.divide(spike_map, occ_map, 
                            out=np.zeros_like(spike_map), 
                            where=occ_map > 0)
        #rate_map = spike_map / occ_map
        rate_map = ndimage.gaussian_filter(rate_map, sigma=1.5)
        
        rate_maps[d_name] = rate_map
        
        if not np.all(np.isnan(rate_map)):
            all_vmax = max(all_vmax, np.nanmax(rate_map))
    
    # 绘制
    for d_id, d_name in direction_names.items():
        row, col = subplot_positions[d_id]
        ax = axes[row, col]
        
        if rate_maps[d_name] is None:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(d_name)
            continue
        
        im = ax.imshow(rate_maps[d_name], origin='lower', cmap='jet',
                      extent=[0, env_size[0], 0, env_size[1]],
                      vmin=0, vmax=rate_maps[d_name].max())
        ax.set_title(f'{d_name} (n={np.sum(direction_labels == d_id) if d_id > 0 else len(positions)})')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        plt.colorbar(im, ax=ax, label='Hz')
    
    # 第6个subplot显示左右差异
    ax = axes[1, 2]
    if rate_maps['Right'] is not None and rate_maps['Left'] is not None:
        diff = rate_maps['Right'] - rate_maps['Left']
        vabs = np.nanmax(np.abs(diff))
        im = ax.imshow(diff, origin='lower', cmap='RdBu_r',
                      extent=[0, env_size[0], 0, env_size[1]],
                      vmin=-vabs, vmax=vabs)
        ax.set_title('Right - Left')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        plt.colorbar(im, ax=ax, label='ΔHz')
    else:
        ax.text(0.5, 0.5, 'Cannot compute\ndifference', transform=ax.transAxes,
               ha='center', va='center')
        ax.set_title('Right - Left')
    
    plt.tight_layout()
    
    if save_path:
        #plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ========== 演示代码 ==========

def demo():
    """演示位置细胞相位差计算"""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("位置细胞相位差计算演示（基于速度方向分类）")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成模拟轨迹（有方向性的随机游走）
    env_size = (100, 100)
    duration = 1000  # 秒
    dt = 0.01
    n_samples = int(duration / dt)
    
    trace_10=np.load(r'D:\References\codings\Remapping\rescale_10\traj_1010cm_dt10_1000s_2_%d.npy'%(1),allow_pickle=True).item()    
    
    positions=trace_10['pos'][::10]
    positions*=100
    
    
    
    # 计算速度方向
    velocities = _compute_velocity(positions, dt)
    
    # 添加方向依赖的相位差
    phase_shift = 5.0  # cm
    spike_train = np.load(r'D:\References\codings\Remapping\rescale_10\init_1126\1010init_bp28\1010cm\spikingrp_init_1010cm_norm3261_greo_al05_Th2_e3_051202_5.npy')
    
    spike_train = spike_train[20]
    
    print(f"总spike数: {int(spike_train.sum())}")
    print(f"平均发放率: {spike_train.sum() / duration:.2f} Hz")
    
    # 计算速度和分类
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    direction_labels = _classify_by_velocity(
        velocities, speeds, axis='x', 
        angle_range=30.0, min_speed=10.0
    )
    
    # 可视化速度分类
    print("\n可视化速度方向分类...")
    visualize_velocity_classification(
        positions, velocities, direction_labels, env_size,
        save_path='/mnt/user-data/outputs/velocity_classification_demo.png'
    )
    
    # 可视化各方向rate map
    print("可视化各方向rate map...")
    visualize_direction_rate_maps(
        positions, spike_train, direction_labels, env_size,
        save_path='/mnt/user-data/outputs/direction_rate_maps_demo.png'
    )
    
    # 计算相位差
    print("\n" + "-" * 40)
    print("计算相位差...")
    print("-" * 40)
    
    result = compute_place_cell_phase_shift(
        positions=positions,
        spike_train=spike_train,
        env_size=env_size,
        axis='x',
        angle_range=20.0,
        min_speed=10.0,
        n_bootstrap=100,
        dt=dt
    )
    
    print(f"\n结果:")
    print(f"  检测到的相位差: {result['shift_cm']:.2f} ± {result['shift_std_cm']:.2f} cm")
    print(f"  真实相位差: {phase_shift:.2f} cm")
    print(f"  方向对: {result['direction_pair']}")
    print(f"  各方向样本数: {result['direction_counts']}")
    
    # 测试不同参数
    print("\n" + "-" * 40)
    print("测试不同角度范围参数...")
    print("-" * 40)
    
    for angle_range in [10, 20, 30, 5]:
        result = compute_place_cell_phase_shift(
            positions=positions,
            spike_train=spike_train,
            env_size=env_size,
            axis='x',
            angle_range=angle_range,
            min_speed=10.0,
            n_bootstrap=50,
            dt=dt
        )
        right_n = result['direction_counts']['right']
        left_n = result['direction_counts']['left']
        print(f"  角度范围±{angle_range}°: shift={result['shift_cm']:.2f}cm, "
              f"样本数(R/L)={right_n}/{left_n}")
    
    print("\n演示完成!")
    
    return result


if __name__ == "__main__":
    result = demo()
