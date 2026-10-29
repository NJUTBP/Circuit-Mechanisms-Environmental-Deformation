# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:49:53 2024

@author: 17534
"""

# Compute Field length & scale

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import opexebo
from ratinabox import utils
import cv2
from scipy.signal import correlate2d

def compute_ncc_using_opencv(A, B):
    # Ensure A and B are float32 for OpenCV functions
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    # Compute the normalized cross-correlation using OpenCV's matchTemplate
    result = cv2.matchTemplate(A, B, method=cv2.TM_CCOEFF_NORMED)    
    return result

def normalized_cross_correlation(matrix1, matrix2):
    # 减去均值
    matrix1 = matrix1 - np.mean(matrix1)
    matrix2 = matrix2 - np.mean(matrix2)

    # 计算互相关
    corr = correlate2d(matrix1, matrix2, mode='full')

    # 计算归一化因子
    norm1 = np.sqrt(np.sum(matrix1**2))
    norm2 = np.sqrt(np.sum(matrix2**2))

    # 归一化
    if norm1 * norm2 == 0:
        return corr * 0  # 避免除以零
    else:
        return corr / (norm1 * norm2)

def cal_field_len(spikingtime,pos,ex,bin_size):
    cell_num=len(spikingtime)
    #bin_size = 0.025
    max_len_p=np.zeros((cell_num,4))
    
    for ip in range(cell_num):
        #从放电数据计算ratemap
        rate_timeseries_ = spikingtime[ip][:len(pos)]
        rate_map, zero_bins = utils.bin_data_for_histogramming(
            data=pos,
            extent=ex,
            dx=bin_size,
            weights=rate_timeseries_,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        rate_map=opexebo.general.smooth(rate_map,2)
        #筛除较低的值
        rate_map[rate_map<0.2*rate_map.max()]=0
        rate_map[rate_map<0.1]=0
        if rate_map.max()<=0.1:
            continue
        #计算自相关图
        grid2_autocorr=opexebo.analysis.autocorrelation(rate_map)
        #看有几个位置野
        
        #只计算大于特定阈值的格点数
        grid2_autocorr[grid2_autocorr<0.2]=0
        
        Field_bool=np.ones_like(grid2_autocorr)
        Field_bool[grid2_autocorr<0.2*grid2_autocorr.max()]=0
        
        #按方向计算放电野大小
        max_tot_h=0 #此处为总长度，即两个放电野的长度相加；
        for i in range(rate_map.shape[0]):
            tot_i=Field_bool[i,:].sum()
            if tot_i>max_tot_h:
                max_tot_h=tot_i
                
        max_tot_v=0 #计算竖向长度
        for j in range(rate_map.shape[1]):
            tot_j=Field_bool[:,j].sum()
            if tot_j>max_tot_v:
                max_tot_v=tot_j
        
        max_single_h=0 #计算单个放电野最大长度
        for i in range(Field_bool.shape[0]):
            tot_i=0
            for j in range(Field_bool.shape[1]):
                if Field_bool[i,j]==0 and tot_i==0 :
                    continue
                elif Field_bool[i,j]==0 and tot_i!=0:
                    if tot_i>max_single_h:
                        max_single_h=tot_i
                    tot_i=0
                if Field_bool[i,j]==1:
                    tot_i+=1
                    if j==Field_bool.shape[1]-1:
                        if tot_i>max_single_h:
                            max_single_h=tot_i
        
        max_single_v=0 #计算竖向单个放电野最大长度
        for j in range(Field_bool.shape[1]):
            tot_j=0
            for i in range(Field_bool.shape[0]):
                if Field_bool[i,j]==0 and tot_j==0 :
                    continue
                elif Field_bool[i,j]==0 and tot_j!=0:
                    if tot_j>max_single_v:
                        max_single_v=tot_j
                    tot_j=0
                if Field_bool[i,j]==1:
                    tot_j+=1
                    if i==Field_bool.shape[0]-1:
                        if tot_j>max_single_v:
                            max_single_v=tot_j   
                
        max_len_p[ip]=[max_tot_h,max_tot_v,max_single_h,max_single_v]
        
    return max_len_p

### 位置细胞位置野长度的计算，直接算ratemap中的格点数，不用自相关；
def cal_pfield_len(spikingtime,pos,ex,bin_size):
    cell_num=len(spikingtime)
    #bin_size = 0.025
    max_len_p=np.zeros((cell_num,4))
    
    for ip in range(cell_num):
        #从放电数据计算ratemap
        rate_timeseries_ = spikingtime[ip][:len(pos)]
        rate_map, zero_bins = utils.bin_data_for_histogramming(
            data=pos,
            extent=ex,
            dx=bin_size,
            weights=rate_timeseries_,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        rate_map=opexebo.general.smooth(rate_map,2)
        #筛除较低的值
        rate_map[rate_map<0.2*rate_map.max()]=0
        rate_map[rate_map<0.1]=0
        if rate_map.max()<=0.1:
            continue
        
        #只计算大于特定阈值的格点数
        Field_bool=np.ones_like(rate_map)
        Field_bool[rate_map<0.2*rate_map.max()]=0
        
        #按方向计算放电野大小
        max_tot_h=0 #此处为总长度，即两个放电野的长度相加；
        for i in range(rate_map.shape[0]):
            tot_i=Field_bool[i,:].sum()
            if tot_i>max_tot_h:
                max_tot_h=tot_i
                
        max_tot_v=0 #计算竖向长度
        for j in range(rate_map.shape[1]):
            tot_j=Field_bool[:,j].sum()
            if tot_j>max_tot_v:
                max_tot_v=tot_j
        
        max_single_h=0 #计算单个放电野最大长度
        for i in range(Field_bool.shape[0]):
            tot_i=0
            for j in range(Field_bool.shape[1]):
                if Field_bool[i,j]==0 and tot_i==0 :
                    continue
                elif Field_bool[i,j]==0 and tot_i!=0:
                    if tot_i>max_single_h:
                        max_single_h=tot_i
                    tot_i=0
                if Field_bool[i,j]==1:
                    tot_i+=1
                    if j==Field_bool.shape[1]-1:
                        if tot_i>max_single_h:
                            max_single_h=tot_i
               
        
        max_single_v=0 #计算竖向单个放电野最大长度
        for j in range(Field_bool.shape[1]):
            tot_j=0
            for i in range(Field_bool.shape[0]):
                if Field_bool[i,j]==0 and tot_j==0 :
                    continue
                elif Field_bool[i,j]==0 and tot_j!=0:
                    if tot_j>max_single_v:
                        max_single_v=tot_j
                    tot_j=0
                if Field_bool[i,j]==1:
                    tot_j+=1
                    if i==Field_bool.shape[0]-1:
                        if tot_j>max_single_v:
                            max_single_v=tot_j
                            
        max_len_p[ip]=[max_tot_h,max_tot_v,max_single_h,max_single_v]
        
    return max_len_p

        
### 
#计算网格间距，最主要的是对放电野的处理，平滑，阈值，还有自相关图的处理，最后其实可以手算
#计算极大值，计算极大值位置与零点之间的距离，取前六个的平均值

### 改成计算rescale factor？
#拉伸原放电野，调包 cv2 ///将放电轨迹全部乘以对应系数再重新计算放电野？
def cal_rescale_factor(rate_map_f,rate_map_n,shift_bin): #熟悉的环境为方形，左右拉伸
    min_bins=np.min((np.min(rate_map_n.shape),np.min(rate_map_f.shape)))-20 #bin_size=0.025
    max_bins=np.max((np.max(rate_map_n.shape),np.max(rate_map_f.shape)))+20
    fam_bins=rate_map_f.shape[0]
    rescale_f=1
    corr_max=0
    id_lr=1
    for i in range(min_bins,max_bins+1):
        rate_map_rs=cv2.resize(rate_map_f, (0, 0),fx=i/fam_bins,fy=1)
        #minor_bins=np.min((i,rate_map_n.shape[1]))
        #corr_i=compute_ncc_using_opencv(rate_map_rs, rate_map_n).max()
        corr_=compute_ncc_using_opencv(rate_map_rs, rate_map_n)
        corr_i=np.max([corr_[0,0:shift_bin],corr_[0,-shift_bin:]]) #好奇怪的调参；；；；
        #corr_i=np.max([corr_[0,0],corr_[0,-1]]) #好奇怪的调参；；；；
        id_lr=np.sign(corr_[0,0]-corr_[0,-1])
        
        #corr_i=normalized_cross_correlation(rate_map_rs, rate_map_n).max() 
        #corr_l=np.corrcoef(rate_map_rs[:,:minor_bins].flatten(),rate_map_n[:,:minor_bins].flatten())[0,1]
        #corr_r=np.corrcoef(rate_map_rs[:,-minor_bins:].flatten(),rate_map_n[:,-minor_bins:].flatten())[0,1]
        #corr_i=np.max((corr_l,corr_r))
        if corr_i>corr_max:
            corr_max=corr_i
            rescale_f=i/fam_bins
            #id_lr=np.sign(corr_r-corr_l)
    return rescale_f,corr_max,id_lr

def cal_grid_rescale_factor(spikingtimef,posf,exf,spikingtimen,posn,exn,shift_bin):
    #bin_size = 0.0125
    bin_size=0.02
    cell_num=len(spikingtimef)
    rescale_gc=np.ones(cell_num)
    corr_rs=np.zeros(cell_num)
    id_lr=np.zeros(cell_num)
    for ip in range(cell_num):
        #从放电数据计算ratemap
        rate_timeseries_f = spikingtimef[ip][:len(posf)]
        rate_mapf, zero_bins = utils.bin_data_for_histogramming(
            data=posf,
            extent=exf,
            dx=bin_size,
            weights=rate_timeseries_f,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        #rate_mapf=opexebo.general.smooth(rate_mapf,2)
        #筛除较低的值
        #rate_mapf[rate_mapf<0.2*rate_mapf.max()]=0
        #rate_mapf[rate_mapf<0.1]=0
        
        if rate_mapf.max()<=0.1:
            continue
        
        rate_timeseries_n = spikingtimen[ip][:len(posn)]
        rate_mapn, zero_bins = utils.bin_data_for_histogramming(
            data=posn,
            extent=exn,
            dx=bin_size,
            weights=rate_timeseries_n,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        #rate_mapn=opexebo.general.smooth(rate_mapn,2)
        #筛除较低的值
        #rate_mapn[rate_mapn<0.2*rate_mapn.max()]=0
        #rate_mapn[rate_mapn<0.1]=0
        
        if rate_mapn.max()<=0.1:
            continue
        
        rescale_gc[ip],corr_rs[ip],id_lr[ip]=cal_rescale_factor(rate_mapf,rate_mapn,shift_bin)
        
    return rescale_gc,corr_rs,id_lr

# 只算一边，并且不算未走过的点
def cal_grid_rescale_factor_lr(spikingtimef,posf,exf,spikingtimen,posn,exn,lr):
    bin_size = 0.025
    cell_num=len(spikingtimef)
    rescale_gc=np.ones(cell_num)
    corr_rs=np.zeros(cell_num)
    for ip in range(cell_num):
        #从放电数据计算ratemap
        rate_timeseries_f = spikingtimef[ip][:len(posf)]
        rate_mapf, zero_binsf = utils.bin_data_for_histogramming(
            data=posf,
            extent=exf,
            dx=bin_size,
            weights=rate_timeseries_f,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        rate_mapf=opexebo.general.smooth(rate_mapf,2)
        #筛除较低的值
        rate_mapf[rate_mapf<0.2*rate_mapf.max()]=0
        rate_mapf[rate_mapf<0.1]=0
        if rate_mapf.max()<=0.1:
            continue
        
        rate_timeseries_n = spikingtimen[ip][:len(posn)]
        rate_mapn, zero_binsn = utils.bin_data_for_histogramming(
            data=posn,
            extent=exn,
            dx=bin_size,
            weights=rate_timeseries_n,
            norm_by_bincount=True,
            return_zero_bins=True,
        )
        #对放电野平滑
        rate_mapn=opexebo.general.smooth(rate_mapn,2)
        #筛除较低的值
        rate_mapn[rate_mapn<0.2*rate_mapn.max()]=0
        rate_mapn[rate_mapn<0.1]=0
        if rate_mapn.max()<=0.1:
            continue
        
        ### 能不能只计算都走过的点的相关性
        zero_binsn=np.logical_not(zero_binsn)
        min_bins=np.min((np.min(rate_mapn.shape),np.min(rate_mapf.shape)))-20 #bin_size=0.025
        max_bins=np.max((np.max(rate_mapn.shape),np.max(rate_mapf.shape)))+20
        fam_bins=rate_mapf.shape[0]
        rescale_f=1
        corr_max=0
        for i in range(min_bins,max_bins+1):
            rate_map_rs=cv2.resize(rate_mapf, (0, 0),fx=i/fam_bins,fy=1)
            minor_bins=np.min((i,rate_mapn.shape[1]))
            corr_l=np.corrcoef(rate_map_rs[:,:minor_bins][zero_binsn[:,:minor_bins]].flatten(),rate_mapn[:,:minor_bins][zero_binsn[:,:minor_bins]].flatten())[0,1]
            corr_r=np.corrcoef(rate_map_rs[:,-minor_bins:][zero_binsn[:,-minor_bins:]].flatten(),rate_mapn[:,-minor_bins:][zero_binsn[:,-minor_bins:]].flatten())[0,1]
            if lr==0:
                corr_i=corr_l
            elif lr==1:
                corr_i=corr_r
            if corr_i>corr_max:
                corr_max=corr_i
                rescale_f=i/fam_bins
        
        rescale_gc[ip]=rescale_f
        corr_rs[ip]=corr_max
    return rescale_gc,corr_rs


    
'''
spikingratep_90=np.load('spikingratep_7090cm_1_3.npy')
trace_90=trace=np.load('traj_7090cm_Ag_1000s_1.npy',allow_pickle=True).item()
Pos_90=trace_90['pos']
spikingratep_70=np.load('spikingratep_70cm_2_1_3.npy')
trace_70=trace=np.load('traj_70cm_2_dt10_1000s_1_3.npy',allow_pickle=True).item()
Pos_70=trace['pos']
ex_70=[0,0.7,0,0.7]
ex_90=[0,0.9,0,0.7]
lenp_90=cal_pfield_len(spikingratep_90,Pos_90,ex_90)
lenp_70=cal_pfield_len(spikingratep_70,Pos_70,ex_70)
rescale_p=[]
#rescale_p=np.zeros(100)
rescale_p2=[] #看竖向的分布
lenp_90[lenp_90<=3]=0
lenp_70[lenp_70<=3]=0
for i in range(100):
    if lenp_90[i,0]==0 and lenp_70[i,0]==0:
        continue
    rescale_p.append(lenp_90[i,0]/lenp_70[i,0])
    #rescale_p[i]=lenp_90[i,0]/lenp_70[i,0]
    
for i in range(100):
    if lenp_90[i,1]==0 or lenp_70[i,1]==0:
        continue
    rescale_p2.append(lenp_90[i,1]/lenp_70[i,1])

plt.figure()
plt.scatter(np.arange(len(rescale_p2)),rescale_p2)

'''

'''
spikingrategc1_70=np.load(r'D:\References\codings\Remapping\square\W_bp_only\spikingr_init\spingrgc1_init_70cm_2_1000s_2.npy')
#spikingrategc2_70=np.load(r'spikingrate_rescale\spikingrategc2_7070cm_100_gp2pg02_1.npy')
trace_70=np.load(r'D:\References\codings\Remapping\square\traj_70cm_2_dt10_1000s_1_1.npy',allow_pickle=True).item()
Pos_70=trace_70['pos']
ex_70=[0,0.7,0,0.7]

spikingrategc1_80=np.load(r'D:\References\codings\Remapping\rescale_70\7080cm\init\spikingrgc1_init_7080cm_1000s_1.npy')
#spikingrategc2_80=np.load(r'spikingrate_rescale\spikingrategc2_7080cm_100_gp2pg02_1.npy')
trace_80=np.load(r'D:\References\codings\Remapping\rescale_70\traj_7080cm_dt10_1000s_1_1.npy',allow_pickle=True).item()
Pos_80=trace_80['pos']
ex_80=[0,0.8,0,0.7]

spikingrategc1_90=np.load(r'spikingrate_rescale\spikingrategc1_7090cm_100_gp2pg02_1.npy')
#spikingrategc2_90=np.load(r'spikingrate_rescale\spikingrategc2_7090cm_100_gp2pg02_1.npy')
trace_90=np.load('traj_7090cm_dt10_1000s_1.npy',allow_pickle=True).item()
Pos_90=trace_90['pos']
ex_90=[0,0.9,0,0.7]

spikingrategc1_10=np.load(r'spikingrate_rescale\spikingrategc1_70100cm_100_gp2pg02_1.npy')
#spikingrategc2_10=np.load(r'spikingrate_rescale\spikingrategc2_70100cm_100_gp2pg02_1.npy')
trace_10=np.load('traj_70100cm_dt10_1000s_1.npy',allow_pickle=True).item()
Pos_10=trace_10['pos']
ex_10=[0,1,0,0.7]

rescale_gc80,corr_rs80=cal_grid_rescale_factor(spikingrategc1_70,Pos_70,ex_70,spikingrategc1_80,Pos_80,ex_80)
rescale_gc280,corr_rs280=cal_grid_rescale_factor(spikingrategc2_70,Pos_70,ex_70,spikingrategc2_80,Pos_80,ex_80)
rescale_gc90,corr_rs90=cal_grid_rescale_factor(spikingrategc1_70,Pos_70,ex_70,spikingrategc1_90,Pos_90,ex_90)
rescale_gc290,corr_rs290=cal_grid_rescale_factor(spikingrategc2_70,Pos_70,ex_70,spikingrategc2_90,Pos_90,ex_90)
rescale_gc10,corr_rs10=cal_grid_rescale_factor(spikingrategc1_70,Pos_70,ex_70,spikingrategc1_10,Pos_10,ex_10)
rescale_gc210,corr_rs210=cal_grid_rescale_factor(spikingrategc2_70,Pos_70,ex_70,spikingrategc2_10,Pos_10,ex_10)

rescale_gc,corr_rs=cal_grid_rescale_factor(spikingrategc1_70,Pos_70,ex_70,spikingrategc1_10,Pos_10,ex_10)
plt.figure()
plt.scatter(np.arange(100),rescale_gc210)
plt.ylabel('Rescale factor')

rescale_gc,corr_rs=cal_grid_rescale_factor_lr(spikingrategc1_70,Pos_70,ex_70,spikingrategc1_80[:,:-1][:,traj_bool_quad[0]],Pos_80[traj_bool_quad[0]],ex_80,0)
'''

