# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:40:11 2025

@author: 17534
"""

import numpy as np
from ratinabox import utils
import opexebo

def cal_ratemap(spikingtime,pos,ex,bin_size):
    cell_num=len(spikingtime)
    x_bins=int(np.ceil((ex[1]-ex[0])/bin_size))
    y_bins=int(np.ceil((ex[3]-ex[2])/bin_size))
    rate_map_all=np.zeros((cell_num,y_bins,x_bins))
    for ip in range(cell_num):
        rate_timeseries = spikingtime[ip][:len(pos)]
        rate_map, zero_bins = utils.bin_data_for_histogramming(
            data=pos,
            extent=ex,
            dx=bin_size,
            weights=rate_timeseries,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        rate_map=opexebo.general.smooth(rate_map,1.5)
        #筛除较低的值
        rate_map[rate_map<0.2*rate_map.max()]=0
        rate_map[rate_map<0.1]=0
        rate_map_all[ip]=rate_map
    return rate_map_all

def cal_grid_scale(rate_map): #仅计算一个细胞的放电野

    cell_num=len(rate_map)
    grid_score=np.zeros(cell_num)
    grid_scale=np.zeros(cell_num)
    for ip in range(cell_num):
        
        auto_corr=opexebo.analysis.autocorrelation(rate_map[ip])
        grid_stats=opexebo.analysis.grid_score(auto_corr)
        grid_score[ip]=grid_stats[0]
        grid_scale[ip]=grid_stats[1]['grid_spacing']
        
    return grid_score,grid_scale

def cal_grid_scale_single(rate_map): #仅计算一个细胞的放电野

        
    auto_corr=opexebo.analysis.autocorrelation(rate_map)
    grid_stats=opexebo.analysis.grid_score(auto_corr)
    grid_score=grid_stats[0]
    grid_scale=grid_stats[1]['grid_spacing']
        
    return grid_score,grid_scale
