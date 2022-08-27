import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
if Station == 2:num_configs = 6
if Station == 3:num_configs = 7
num_ants = 16
if Station == 2:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.07, 0.09, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_rf_04[:,1] = np.array([0.07, 0.17, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.09, 0.13, 0.09, 0.11, 0.11, 0.11, 0.09, 1000], dtype = float)
            cw_rf_04[:,2] = np.array([0.07, 0.09, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.07, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.07, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.09, 0.11, 0.11, 0.15, 0.11, 0.11, 0.11, 0.11, 0.13, 0.13, 0.13, 0.25, 0.13, 0.09, 0.13, 1000], dtype = float)
            cw_rf_025[:,1] = np.array([0.11, 0.13, 0.11, 0.13, 0.11, 0.11, 0.25, 0.11, 0.13, 0.13, 0.13, 0.21, 0.13, 0.13, 0.13, 1000], dtype = float)
            cw_rf_025[:,2] = np.array([0.17, 0.17, 0.17, 0.23, 0.15, 0.17, 0.15, 0.15, 0.21, 0.19, 0.19, 0.33, 0.19, 0.17, 0.21, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.11, 0.11, 0.11, 0.11, 0.09, 0.11, 0.05, 0.09, 0.15, 0.13, 0.13, 0.29, 0.11, 0.13, 0.13, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.13, 0.15, 0.11, 0.13, 0.11, 0.11, 0.21, 0.11, 0.15, 0.15, 0.15, 0.21, 0.11, 0.13, 0.15, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.09, 0.11, 0.09, 0.11, 0.09, 0.09, 0.09, 0.09, 0.13, 0.11, 0.11, 0.19, 0.11, 0.11, 0.11, 1000], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.03, 0.09, 0.03, 0.05, 0.07, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.05, 0.05, 0.05, 0.03, 1000], dtype = float)
            cw_rf_0125[:,1] = np.array([0.03, 0.07, 0.05, 0.09, 0.23, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.05, 0.05, 0.07, 0.03, 1000], dtype = float)
            cw_rf_0125[:,2] = np.array([0.05, 0.11, 0.07, 0.09, 0.15, 0.07, 0.07, 0.07, 0.09, 0.05, 0.03, 0.07, 0.07, 0.11, 0.05, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.03, 0.03, 0.03, 0.03, 0.23, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.03, 0.05, 0.03, 0.03, 0.29, 0.05, 0.05, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.03, 0.03, 0.03, 0.03, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.05, 0.05, 0.03, 0.03, 0.05, 0.03, 0.03, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,1] = np.array([0.07, 0.13, 0.07, 0.07, 0.09, 0.07, 0.05, 0.07, 0.11, 0.09, 0.07, 0.11, 0.09, 0.11, 0.07, 1000], dtype = float)
            cw_cal_04[:,2] = np.array([0.05, 0.09, 0.03, 0.03, 0.05, 0.05, 0.03, 0.05, 0.07, 0.05, 0.05, 0.05, 0.07, 0.07, 0.05, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.05, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.07, 0.07, 0.05, 0.07, 0.07, 0.05, 0.07, 0.07, 0.07, 0.05, 0.05, 0.05, 0.07, 0.05, 0.07, 1000], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.13, 0.11, 0.07, 0.09, 0.07, 0.09, 0.07, 0.07, 0.13, 0.13, 0.13, 0.21, 0.13, 0.11, 0.11, 1000], dtype = float)
            cw_cal_025[:,1] = np.array([0.13, 0.15, 0.13, 0.19, 0.13, 0.11, 0.25, 0.11, 0.21, 0.19, 0.19, 0.29, 0.13, 0.19, 0.17, 1000], dtype = float)
            cw_cal_025[:,2] = np.array([0.13, 0.13, 0.07, 0.11, 0.09, 0.09, 0.07, 0.11, 0.15, 0.15, 0.17, 0.21, 0.19, 0.13, 0.13, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.11, 0.13, 0.09, 0.15, 0.13, 0.09, 0.11, 0.09, 0.17, 0.15, 0.15, 0.31, 0.11, 0.17, 0.13, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.13, 0.13, 0.13, 0.15, 0.11, 0.11, 0.17, 0.11, 0.15, 0.15, 0.13, 0.23, 0.11, 0.15, 0.17, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.11, 0.15, 0.13, 0.15, 0.11, 0.09, 0.11, 0.11, 0.17, 0.15, 0.15, 0.23, 0.11, 0.15, 0.17, 1000], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.03, 0.05, 0.03, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.05, 1000], dtype = float)
            cw_cal_0125[:,1] = np.array([0.05, 0.09, 0.05, 0.07, 0.11, 0.05, 0.05, 0.05, 0.07, 0.05, 0.03, 0.07, 0.07, 0.09, 0.05, 1000], dtype = float)
            cw_cal_0125[:,2] = np.array([0.05, 0.05, 0.05, 0.07, 0.07, 0.05, 0.03, 0.05, 0.07, 0.05, 0.03, 0.07, 0.05, 0.09, 0.03, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.05, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.07, 0.07, 0.03, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.03, 0.05, 0.05, 0.03, 0.15, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.03, 0.03, 0.05, 0.03, 0.11, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 1000], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.15, 0.23, 0.15, 0.15, 0.19, 0.15, 0.15, 0.15, 0.21, 0.17, 0.17, 0.21, 0.17, 0.17, 0.19, 1000], dtype = float)
            cw_soft_04[:,1] = np.array([0.17, 0.27, 0.15, 0.15, 0.21, 0.17, 0.15, 0.17, 0.19, 0.21, 0.19, 0.29, 0.21, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,2] = np.array([0.17, 0.25, 0.15, 0.15, 0.21, 0.17, 0.17, 0.17, 0.21, 0.17, 0.23, 0.27, 0.19, 0.19, 0.23, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.15, 0.23, 0.15, 0.15, 0.21, 0.15, 0.15, 0.15, 0.19, 0.17, 0.19, 0.19, 0.21, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.15, 0.23, 0.15, 0.13, 0.19, 0.15, 0.15, 0.15, 0.19, 0.19, 0.21, 0.17, 0.19, 0.17, 0.21, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.11, 0.11, 0.11, 0.11, 0.13, 0.11, 0.11, 0.11, 0.15, 0.11, 0.11, 0.09, 0.17, 0.11, 0.11, 1000], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.25, 0.31, 0.27, 0.35, 0.35, 0.27, 0.21, 0.29, 0.35, 0.39, 0.35, 0.47, 0.29, 0.33, 0.39, 1000], dtype = float)
            cw_soft_025[:,1] = np.array([0.33, 0.37, 0.31, 0.37, 0.37, 0.31, 0.41, 0.31, 0.37, 0.41, 0.35, 0.51, 0.37, 0.39, 0.35, 1000], dtype = float)
            cw_soft_025[:,2] = np.array([0.31, 0.37, 0.31, 0.39, 0.35, 0.31, 0.29, 0.31, 0.37, 0.41, 0.35, 0.51, 0.37, 0.39, 0.35, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.29, 0.35, 0.29, 0.35, 0.31, 0.29, 0.35, 0.27, 0.51, 0.41, 0.41, 0.55, 0.37, 0.37, 0.41, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.31, 0.35, 0.31, 0.37, 0.37, 0.31, 0.43, 0.33, 0.41, 0.39, 0.37, 0.51, 0.31, 0.33, 0.33, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.25, 0.31, 0.23, 0.27, 0.23, 0.23, 0.23, 0.23, 0.45, 0.29, 0.29, 0.43, 0.27, 0.37, 0.37, 1000], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.05, 0.25, 0.15, 0.23, 0.19, 0.15, 0.15, 0.11, 0.17, 0.13, 0.11, 0.13, 0.13, 0.23, 0.15, 1000], dtype = float)
            cw_soft_0125[:,1] = np.array([0.09, 0.33, 0.21, 0.27, 0.27, 0.13, 0.15, 0.13, 0.21, 0.17, 0.13, 0.21, 0.17, 0.25, 0.21, 1000], dtype = float)
            cw_soft_0125[:,2] = np.array([0.07, 0.31, 0.17, 0.21, 0.27, 0.17, 0.17, 0.13, 0.23, 0.15, 0.11, 0.17, 0.15, 0.27, 0.19, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.05, 0.27, 0.15, 0.17, 0.25, 0.17, 0.13, 0.13, 0.19, 0.17, 0.13, 0.23, 0.17, 0.27, 0.19, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.09, 0.23, 0.13, 0.19, 0.25, 0.15, 0.13, 0.13, 0.21, 0.17, 0.13, 0.17, 0.17, 0.27, 0.21, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.05, 0.07, 0.09, 0.13, 0.21, 0.11, 0.11, 0.11, 0.19, 0.11, 0.07, 0.13, 0.13, 0.15, 0.11, 1000], dtype = float)

if Station == 3:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.09, 0.07, 0.07, 0.07, 0.07, 0.11, 0.07, 0.07], dtype = float)
            cw_rf_04[:,1] = np.array([0.11, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07], dtype = float)
            cw_rf_04[:,2] = np.array([0.09, 0.07, 0.07, 1000, 0.05, 0.05, 0.09, 1000, 0.07, 0.07, 0.11, 1000, 0.05, 0.11, 0.13, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.07, 0.05, 0.05, 1000, 0.05, 0.05, 0.09, 1000, 0.05, 0.07, 0.11, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.11, 0.07, 0.07, 1000, 0.07, 0.07, 0.09, 1000, 0.09, 0.09, 0.09, 1000, 0.07, 0.11, 0.09, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.07, 0.07, 0.11, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05, 0.11, 0.13, 0.15, 0.05, 0.11, 0.13, 0.07], dtype = float)
            cw_rf_04[:,6] = np.array([1000, 0.07, 0.05, 0.15, 1000, 0.05, 0.15, 0.03, 1000, 0.07, 0.13, 0.11, 1000, 0.05, 0.11, 0.07], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.17, 0.11, 0.11, 0.11, 0.15, 0.13, 0.15, 0.11, 0.13, 0.15, 0.17, 0.09, 0.15, 0.11, 0.17, 0.11], dtype = float)
            cw_rf_025[:,1] = np.array([0.13, 0.11, 0.11, 0.11, 0.13, 0.13, 0.13, 0.13, 0.15, 0.13, 0.19, 0.13, 0.15, 0.15, 0.17, 0.13], dtype = float)
            cw_rf_025[:,2] = np.array([0.13, 0.07, 0.09, 1000, 0.13, 0.11, 0.09, 1000, 0.13, 0.13, 0.11, 1000, 0.13, 0.15, 0.15, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.17, 0.07, 0.13, 1000, 0.11, 0.11, 0.13, 1000, 0.13, 0.17, 0.11, 1000, 0.15, 0.19, 0.19, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.15, 0.11, 0.15, 1000, 0.15, 0.15, 0.13, 1000, 0.17, 0.15, 0.15, 1000, 0.15, 0.17, 0.17, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.07, 0.13, 0.13, 0.11, 0.09, 0.11, 0.11, 0.11, 0.15, 0.09, 0.09, 0.07, 0.09, 0.13, 0.09, 0.07], dtype = float)
            cw_rf_025[:,6] = np.array([1000, 0.09, 0.11, 0.07, 1000, 0.11, 0.07, 0.09, 1000, 0.11, 0.13, 0.11, 1000, 0.13, 0.15, 0.09], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.05, 0.03, 0.03, 0.03, 0.03, 0.07, 0.07, 0.03, 0.03, 0.05, 0.03, 0.05, 0.05, 0.11, 0.05, 0.03], dtype = float)
            cw_rf_0125[:,1] = np.array([0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03, 0.03, 0.05, 0.03, 0.05, 0.05, 0.09, 0.03, 0.03], dtype = float)
            cw_rf_0125[:,2] = np.array([0.05, 0.07, 0.03, 1000, 0.03, 0.05, 0.07, 1000, 0.03, 0.03, 0.03, 1000, 0.04, 0.07, 0.03, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.09, 0.11, 0.11, 1000, 0.03, 0.09, 0.05, 1000, 0.03, 0.09, 0.09, 1000, 0.04, 0.13, 0.11, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.05, 0.03, 0.03, 1000, 0.03, 0.05, 0.05, 1000, 0.05, 0.05, 0.03, 1003, 0.05, 0.07, 0.05, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.03, 0.11, 0.09, 0.03, 0.03, 0.05, 0.05, 0.03, 0.03, 0.07, 0.09, 0.06, 0.03, 0.07, 0.07, 0.03], dtype = float)
            cw_rf_0125[:,6] = np.array([1000, 0.07, 0.03, 0.11, 1000, 0.03, 0.03, 0.15, 1000, 0.03, 0.03, 0.13, 1000, 0.03, 0.03, 0.15], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.05, 0.05, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07], dtype = float)
            cw_cal_04[:,1] = np.array([0.07, 0.13, 0.07, 0.07, 0.09, 0.07, 0.07, 0.05, 0.09, 0.11, 0.09, 0.07, 0.07, 0.11, 0.07, 0.07], dtype = float)
            cw_cal_04[:,2] = np.array([0.05, 0.05, 0.03, 1000, 0.03, 0.03, 0.05, 1000, 0.07, 0.07, 0.07, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.05, 0.05, 0.03, 1000, 0.03, 0.03, 0.05, 1000, 0.07, 0.07, 0.09, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.07, 0.05, 0.07, 1000, 0.03, 0.03, 0.05, 1000, 0.09, 0.09, 0.07, 1000, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.05, 0.05, 0.03, 0.13, 0.03, 0.03, 0.13, 0.15, 0.05, 0.07, 0.05, 0.07, 0.05, 0.05, 0.09, 0.07], dtype = float)
            cw_cal_04[:,6] = np.array([1000, 0.05, 0.03, 0.11, 1000, 0.03, 0.13, 0.09, 1000, 0.09, 0.11, 0.11, 1000, 0.05, 0.13, 0.07], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.13, 0.09, 0.13, 0.11, 0.07, 0.07, 0.07, 0.09, 0.17, 0.15, 0.17, 0.13, 0.17, 0.15, 0.17, 0.15], dtype = float)
            cw_cal_025[:,1] = np.array([0.15, 0.11, 0.11, 0.11, 0.13, 0.11, 0.13, 0.13, 0.19, 0.13, 0.25, 0.17, 0.17, 0.17, 0.19, 0.19], dtype = float)
            cw_cal_025[:,2] = np.array([0.11, 0.09, 0.11, 1000, 0.09, 0.07, 0.09, 1000, 0.15, 0.13, 0.13, 1000, 0.17, 0.15, 0.17, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.11, 0.09, 0.09, 1000, 0.09, 0.11, 0.07, 1000, 0.17, 0.17, 0.15, 1000, 0.15, 0.15, 0.17, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.15, 0.11, 0.13, 1000, 0.11, 0.09, 0.07, 1000, 0.19, 0.15, 0.17, 1000, 0.17, 0.19, 0.21, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.11, 0.09, 0.13, 0.09, 0.11, 0.11, 0.07, 0.13, 0.17, 0.09, 0.15, 0.17, 0.13, 0.15, 0.15, 0.13], dtype = float)
            cw_cal_025[:,6] = np.array([1000, 0.07, 0.09, 0.05, 1000, 0.11, 0.09, 0.07, 1000, 0.11, 0.13, 0.09, 1000, 0.13, 0.15, 0.09], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.03, 0.03, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03, 0.05, 0.05, 0.09, 0.03, 0.05], dtype = float)
            cw_cal_0125[:,1] = np.array([0.05, 0.03, 0.07, 0.05, 0.03, 0.05, 0.07, 0.03, 0.05, 0.07, 0.03, 0.07, 0.07, 0.09, 0.05, 0.03], dtype = float)
            cw_cal_0125[:,2] = np.array([0.03, 0.05, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.05, 0.03, 1000, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.05, 0.03, 1000, 0.03, 0.07, 0.03, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.05, 0.03, 0.07, 1000, 0.03, 0.03, 0.03, 1000, 0.05, 0.07, 0.03, 1000, 0.07, 0.09, 0.05, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.03, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03, 0.05, 0.05, 0.03, 0.03], dtype = float)
            cw_cal_0125[:,6] = np.array([1000, 0.05, 0.03, 0.13, 1000, 0.05, 0.03, 0.15, 1000, 0.03, 0.03, 0.15, 1000, 0.05, 0.03, 0.19], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.23, 0.17, 0.17, 0.17, 0.17, 0.15, 0.15, 0.13, 0.25, 0.19, 0.17, 0.15, 0.17, 0.19, 0.21, 0.15], dtype = float)
            cw_soft_04[:,1] = np.array([0.35, 0.21, 0.21, 0.19, 0.19, 0.17, 0.15, 0.15, 0.21, 0.23, 0.21, 0.15, 0.19, 0.23, 0.17, 0.15], dtype = float)
            cw_soft_04[:,2] = np.array([0.31, 0.19, 0.19, 1000, 0.17, 0.17, 0.17, 1000, 0.21, 0.21, 0.25, 1000, 0.17, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.21, 0.17, 0.17, 1000, 0.15, 0.15, 0.17, 1000, 0.19, 0.21, 0.27, 1000, 0.15, 0.19, 0.25, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.29, 0.19, 0.19, 1000, 0.17, 0.17, 0.17, 1000, 0.23, 0.19, 0.19, 1000, 0.17, 0.17, 0.19, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.13, 0.13, 0.13, 0.19, 0.11, 0.11, 0.21, 0.11, 0.13, 0.13, 0.15, 0.19, 0.11, 0.15, 0.17, 0.15], dtype = float)
            cw_soft_04[:,6] = np.array([1000, 0.13, 0.11, 0.11, 1000, 0.11, 0.19, 0.09, 1000, 0.15, 0.17, 0.19, 1000, 0.15, 0.21, 0.19], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.27, 0.23, 0.23, 0.23, 0.33, 0.23, 0.31, 0.29, 0.35, 0.27, 0.33, 0.25, 0.33, 0.29, 0.33, 0.27], dtype = float)
            cw_soft_025[:,1] = np.array([0.33, 0.29, 0.29, 0.27, 0.35, 0.33, 0.39, 0.35, 0.37, 0.39, 0.47, 0.35, 0.37, 0.39, 0.41, 0.37], dtype = float)
            cw_soft_025[:,2] = np.array([0.37, 0.29, 0.31, 1000, 0.35, 0.31, 0.29, 1000, 0.39, 0.33, 0.37, 1000, 0.37, 0.35, 0.35, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.31, 0.25, 0.29, 1000, 0.31, 0.31, 0.25, 1000, 0.35, 0.33, 0.29, 1000, 0.35, 0.33, 0.41, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.41, 0.27, 0.29, 1000, 0.35, 0.31, 0.31, 1000, 0.29, 0.31, 0.35, 1000, 0.37, 0.35, 0.41, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.23, 0.17, 0.25, 0.15, 0.25, 0.23, 0.17, 0.21, 0.31, 0.23, 0.31, 0.27, 0.29, 0.27, 0.29, 0.23], dtype = float)
            cw_soft_025[:,6] = np.array([1000, 0.15, 0.23, 0.09, 1000, 0.21, 0.15, 0.17, 1000, 0.23, 0.25, 0.19, 1000, 0.27, 0.27, 0.17], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.11, 0.09, 0.09, 0.09, 0.09, 0.13, 0.15, 0.07, 0.11, 0.11, 0.07, 0.13, 0.11, 0.21, 0.09, 0.05], dtype = float)
            cw_soft_0125[:,1] = np.array([0.17, 0.09, 0.11, 0.11, 0.13, 0.17, 0.15, 0.11, 0.15, 0.15, 0.15, 0.17, 0.19, 0.29, 0.13, 0.11], dtype = float)
            cw_soft_0125[:,2] = np.array([0.15, 0.11, 0.11, 1000, 0.11, 0.17, 0.13, 1000, 0.09, 0.15, 0.13, 1000, 0.15, 0.23, 0.13, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.19, 0.11, 0.11, 1000, 0.11, 0.17, 0.13, 1000, 0.17, 0.15, 0.13, 1000, 0.17, 0.25, 0.13, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.13, 0.11, 0.11, 1000, 0.13, 0.17, 0.15, 1000, 0.15, 0.13, 0.15, 1003, 0.15, 0.21, 0.13, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.07, 0.07, 0.11, 0.07, 0.05, 0.13, 0.09, 0.05, 0.07, 0.11, 0.09, 0.09, 0.07, 0.19, 0.11, 0.05], dtype = float)
            cw_soft_0125[:,6] = np.array([1000, 0.07, 0.13, 0.27, 1000, 0.13, 0.07, 0.37, 1000, 0.09, 0.05, 0.33, 1000, 0.15, 0.09, 0.31], dtype = float)


trig = int(sys.argv[2])

count_i = int(sys.argv[3])
count_f = int(sys.argv[4])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_val/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

# wb map
md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013 = int(md_2013_r.timestamp())
md_2020 = datetime(2020, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020 = int(md_2020_r.timestamp())
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_bins_i = unix_min_bins[0]
min_in_day = 24 * 60
hrs_in_days = np.arange(min_in_day) / 60
del md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020 

#output
ratio_04_map = np.full((len(unix_min_bins[:-1]), num_ants), 0, dtype = int)
ratio_04_pass_map = np.copy(ratio_04_map)
ratio_04_cut_map = np.copy(ratio_04_map)
ratio_025_map = np.copy(ratio_04_map)
ratio_025_pass_map = np.copy(ratio_04_map)
ratio_025_cut_map = np.copy(ratio_04_map)
ratio_0125_map = np.copy(ratio_04_map)
ratio_0125_pass_map = np.copy(ratio_04_map)
ratio_0125_cut_map = np.copy(ratio_04_map)
print('map array done!')

ratio_bins = np.linspace(0, 100, 50 + 1, dtype = int)
ratio_bin_center = ((ratio_bins[1:] + ratio_bins[:-1]) / 2).astype(int)
ratio_bin_len = len(ratio_bin_center)
ratio_04_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = int)
ratio_04_pass_hist = np.copy(ratio_04_hist)
ratio_04_cut_hist = np.copy(ratio_04_hist)
ratio_025_hist = np.copy(ratio_04_hist)
ratio_025_pass_hist = np.copy(ratio_04_hist)
ratio_025_cut_hist = np.copy(ratio_04_hist)
ratio_0125_hist = np.copy(ratio_04_hist)
ratio_0125_pass_hist = np.copy(ratio_04_hist)
ratio_0125_cut_hist = np.copy(ratio_04_hist)
print('hist array done!')

def get_max_2d(x, y, x_bins):

    xy = np.histogram2d(x, y, bins = (x_bins, ratio_bins))[0].astype(int)
    xy[xy != 0] = 1
    xy *= ratio_bin_center[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

count_ff = count_i + count_f

if trig == 0:
    cw_04 = np.copy(cw_rf_04) * 100
    cw_025 = np.copy(cw_rf_025) * 100
    cw_0125 = np.copy(cw_rf_0125) * 100
if trig == 1:
    cw_04 = np.copy(cw_cal_04) * 100
    cw_025 = np.copy(cw_cal_025) * 100
    cw_0125 = np.copy(cw_cal_0125) * 100
if trig == 2:
    cw_04 = np.copy(cw_soft_04) * 100
    cw_025 = np.copy(cw_soft_025) * 100
    cw_0125 = np.copy(cw_soft_0125) * 100

ant_c = 1

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['unix_time'][:]
    trig_type = hf['trig_type'][:]
    time_bins = hf['time_bins'][:]
    unix_idx = ((time_bins + 0.5 - unix_min_bins_i) // 60).astype(int)[:-1]
    sub_r = hf['sub_ratios'][:]
    sub_r[:, :, trig_type != trig] = np.nan
    sub_r *= 100
    sub_04 = sub_r[2]
    sub_025 = sub_r[1]
    sub_0125 = sub_r[0]
    del hf, trig_type

    cut_04_idx = np.count_nonzero(sub_04 > cw_04[:, g_idx][:, np.newaxis], axis = 0) > ant_c
    pass_04_idx = ~cut_04_idx
    cut_025_idx = np.count_nonzero(sub_025 > cw_025[:, g_idx][:, np.newaxis], axis = 0) > ant_c
    pass_025_idx = ~cut_025_idx
    cut_0125_idx = np.count_nonzero(sub_0125 > cw_0125[:, g_idx][:, np.newaxis], axis = 0) > ant_c
    pass_0125_idx = ~cut_0125_idx

    sub_04_pass = np.copy(sub_04)
    sub_04_pass[:, cut_04_idx] = np.nan
    sub_04_cut = np.copy(sub_04)
    sub_04_cut[:, pass_04_idx] = np.nan
    sub_025_pass = np.copy(sub_025)
    sub_025_pass[:, cut_025_idx] = np.nan
    sub_025_cut = np.copy(sub_025)
    sub_025_cut[:, pass_025_idx] = np.nan
    sub_0125_pass = np.copy(sub_0125)
    sub_0125_pass[:, cut_0125_idx] = np.nan
    sub_0125_cut = np.copy(sub_0125)
    sub_0125_cut[:, pass_0125_idx] = np.nan
    del cut_04_idx, pass_04_idx, cut_025_idx, pass_025_idx, cut_0125_idx, pass_0125_idx

    for ant in range(num_ants):
        ratio_04_hist[:, ant, g_idx] += np.histogram(sub_04[ant], bins = ratio_bins)[0].astype(int) 
        ratio_04_pass_hist[:, ant, g_idx] += np.histogram(sub_04_pass[ant], bins = ratio_bins)[0].astype(int) 
        ratio_04_cut_hist[:, ant, g_idx] += np.histogram(sub_04_cut[ant], bins = ratio_bins)[0].astype(int) 
        ratio_025_hist[:, ant, g_idx] += np.histogram(sub_025[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_pass_hist[:, ant, g_idx] += np.histogram(sub_025_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_cut_hist[:, ant, g_idx] += np.histogram(sub_025_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_hist[:, ant, g_idx] += np.histogram(sub_0125[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_pass_hist[:, ant, g_idx] += np.histogram(sub_0125_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_cut_hist[:, ant, g_idx] += np.histogram(sub_0125_cut[ant], bins = ratio_bins)[0].astype(int)
    
        ratio_04_map[unix_idx, ant] = get_max_2d(unix_time, sub_04[ant], time_bins)
        ratio_04_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_04_pass[ant], time_bins)
        ratio_04_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_04_cut[ant], time_bins)
        ratio_025_map[unix_idx, ant] = get_max_2d(unix_time, sub_025[ant], time_bins)
        ratio_025_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_025_pass[ant], time_bins)
        ratio_025_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_025_cut[ant], time_bins)
        ratio_0125_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125[ant], time_bins)
        ratio_0125_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125_pass[ant], time_bins)
        ratio_0125_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125_cut[ant], time_bins)

    del g_idx, unix_time, time_bins, unix_idx, sub_r, sub_04, sub_025, sub_0125
    del sub_04_pass, sub_04_cut, sub_025_pass, sub_025_cut, sub_0125_pass, sub_0125_cut

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
ratio_04_map = np.reshape(ratio_04_map, (-1, min_in_day, num_ants))
ratio_04_pass_map = np.reshape(ratio_04_pass_map, (-1, min_in_day, num_ants))
ratio_04_cut_map = np.reshape(ratio_04_cut_map, (-1, min_in_day, num_ants))
ratio_025_map = np.reshape(ratio_025_map, (-1, min_in_day, num_ants))
ratio_025_pass_map = np.reshape(ratio_025_pass_map, (-1, min_in_day, num_ants))
ratio_025_cut_map = np.reshape(ratio_025_cut_map, (-1, min_in_day, num_ants))
ratio_0125_map = np.reshape(ratio_0125_map, (-1, min_in_day, num_ants))
ratio_0125_pass_map = np.reshape(ratio_0125_pass_map, (-1, min_in_day, num_ants))
ratio_0125_cut_map = np.reshape(ratio_0125_cut_map, (-1, min_in_day, num_ants))
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)
del min_in_day, unix_min_bins_i

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_Test_Cut_A{Station}_{trig}_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_map', data=ratio_04_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_map', data=ratio_04_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_map', data=ratio_04_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_map', data=ratio_025_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_map', data=ratio_025_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_map', data=ratio_025_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_map', data=ratio_0125_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_map', data=ratio_0125_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_map', data=ratio_0125_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_hist', data=ratio_04_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_hist', data=ratio_04_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_hist', data=ratio_04_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_hist', data=ratio_025_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_hist', data=ratio_025_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_hist', data=ratio_025_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_hist', data=ratio_0125_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_hist', data=ratio_0125_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_hist', data=ratio_0125_cut_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






