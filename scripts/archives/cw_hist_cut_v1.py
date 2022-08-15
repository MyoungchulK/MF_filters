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
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
#d_type = 'all_002'
d_type = 'wb_002'
#d_type = 'ha_002'
#d_type = 'ha2_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_04/*'
#d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
#d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
hf = h5py.File(d_list[0], 'r')
ratio_bin_center = hf['ratio_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
del hf

ratio_len = len(ratio_bin_center)

if Station == 2:
    g_dim = 6
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([0.08, 0.08, 0.08, 0.08,  0.1, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06, 0.08, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr[:,1] = np.array([0.08, 0.08, 0.08, 0.06,  0.1, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06, 0.08, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr[:,2] = np.array([0.08, 0.06, 0.06, 0.06,  0.1, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 1000], dtype = float)
    cut_arr[:,3] = np.array([0.08, 0.06, 0.06, 0.06,  0.1, 0.06, 0.06, 0.08, 0.04, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 1000], dtype = float)
    cut_arr[:,4] = np.array([0.06, 0.06, 0.06, 0.06,  0.1, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 1000], dtype = float)
    cut_arr[:,5] = np.array([0.06, 0.06, 0.06, 0.06, 0.08, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 1000], dtype = float)
    """
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([0.12, 0.12, 0.12, 0.12,  0.1,  0.1,  0.1,  0.1, 0.14, 0.14, 0.14, 0.18, 0.12, 0.12, 0.14, 1000], dtype = float)
    cut_arr[:,1] = np.array([0.12, 0.12,  0.1, 0.14,  0.1,  0.1, 0.18,  0.1, 0.14, 0.14, 0.14, 0.18, 0.12, 0.12, 0.14, 1000], dtype = float)
    cut_arr[:,2] = np.array([ 0.1,  0.1,  0.1, 0.12,  0.1,  0.1,  0.1,  0.1, 0.12, 0.12, 0.14, 0.18, 0.12, 0.12, 0.12, 1000], dtype = float)
    cut_arr[:,3] = np.array([ 0.1,  0.1,  0.1,  0.1, 0.08, 0.08, 0.12, 0.08, 0.12, 0.12, 0.12, 0.22,  0.1,  0.1,  0.1, 1000], dtype = float)
    cut_arr[:,4] = np.array([ 0.1,  0.1, 0.08,  0.1, 0.08, 0.08,  0.1, 0.08, 0.12, 0.12,  0.1, 0.14,  0.1,  0.1,  0.1, 1000], dtype = float)
    cut_arr[:,5] = np.array([ 0.1,  0.1, 0.08,  0.1, 0.08, 0.08, 0.08, 0.08, 0.12,  0.1,  0.1, 0.14,  0.1,  0.1,  0.1, 1000], dtype = float)
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([0.06, 0.12, 0.06,  0.1,  0.1, 0.08, 0.08, 0.06, 0.08, 0.08, 0.06, 0.08, 0.06, 0.14, 0.08, 1000], dtype = float)
    cut_arr[:,1] = np.array([0.06, 0.14, 0.06, 0.14, 0.12, 0.06, 0.08, 0.06, 0.08, 0.08, 0.06, 0.06, 0.06,  0.1, 0.08, 1000], dtype = float)
    cut_arr[:,2] = np.array([0.04, 0.12, 0.06, 0.08,  0.1, 0.08, 0.08, 0.06, 0.08, 0.08, 0.06, 0.06, 0.06, 0.12, 0.06, 1000], dtype = float)
    cut_arr[:,3] = np.array([0.06,  0.1, 0.04, 0.08, 0.16, 0.08, 0.06, 0.06, 0.08, 0.06, 0.06, 0.06, 0.06,  0.1, 0.06, 1000], dtype = float)
    cut_arr[:,4] = np.array([0.04,  0.1, 0.04, 0.06, 0.16, 0.08, 0.06, 0.06, 0.08, 0.06, 0.04, 0.06, 0.04,  0.1, 0.06, 1000], dtype = float)
    cut_arr[:,5] = np.array([0.04, 0.08, 0.04, 0.06,  0.1, 0.06, 0.06, 0.06, 0.08, 0.06, 0.04, 0.06, 0.04, 0.08, 0.06, 1000], dtype = float)

    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 1000], dtype = float)
    cut_arr_cut[:,1] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr_cut[:,2] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
    """
    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([0.16, 0.16, 0.16, 0.16, 0.14, 0.14, 0.14, 0.14, 0.18,  0.2, 0.16, 0.24, 0.16, 0.16, 0.16, 1000], dtype = float)
    cut_arr_cut[:,1] = np.array([0.16, 0.16, 0.16,  0.2, 0.14, 0.12, 0.24, 0.14,  0.2,  0.2, 0.18, 0.24, 0.16, 0.18, 0.16, 1000], dtype = float)
    cut_arr_cut[:,2] = np.array([0.14, 0.16, 0.14, 0.14,  0.1,  0.1, 0.14, 0.14, 0.18, 0.18, 0.18, 0.26, 0.16, 0.16, 0.16, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.16, 0.26, 0.14, 0.14, 0.14, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.14, 0.18, 0.14, 0.12, 0.14, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([0.12, 0.12,  0.1, 0.12,  0.1,  0.1,  0.1,  0.1, 0.14, 0.16, 0.14, 0.18, 0.12, 0.12, 0.12, 1000], dtype = float)
    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([0.06, 0.18, 0.06, 0.14, 0.12, 0.08, 0.08, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08,  0.2,  0.1, 1000], dtype = float)
    cut_arr_cut[:,1] = np.array([0.06,  0.2, 0.06, 0.18,  0.2, 0.08,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
    cut_arr_cut[:,2] = np.array([0.08, 0.16, 0.06, 0.12, 0.14,  0.1,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.08, 0.14, 0.04, 0.08, 0.22, 0.08, 0.06, 0.06,  0.1,  0.1, 0.08, 0.08, 0.06, 0.14, 0.08, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.04, 0.14, 0.04, 0.06, 0.22, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.06, 0.06, 0.14, 0.06, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([0.04, 0.08, 0.04, 0.06, 0.14, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.08, 0.06, 0.12, 0.08, 1000], dtype = float)

if Station == 3:
    g_dim = 7
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([ 0.1, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04], dtype = float)
    cut_arr[:,1] = np.array([ 0.1, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06], dtype = float)
    cut_arr[:,2] = np.array([ 0.1, 0.06, 0.06, 1000, 0.06, 0.06, 0.08, 1000, 0.06, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr[:,3] = np.array([0.08, 0.06, 0.06, 1000, 0.06, 0.06,  0.1, 1000, 0.06, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr[:,4] = np.array([0.12, 0.08, 0.08, 1000, 0.08, 0.06, 0.08, 1000, 0.08, 0.06, 0.06, 1000, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr[:,5] = np.array([0.06, 0.06, 0.06,  0.1, 0.06, 0.06, 0.12, 0.06, 0.06, 0.04, 0.04, 0.06, 0.04, 0.04, 0.06, 0.04], dtype = float)
    cut_arr[:,6] = np.array([1000, 0.06, 0.06, 0.06, 1000, 0.06, 0.12, 0.04, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.06, 0.04], dtype = float)
    """
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([ 0.1,  0.1,  0.1,  0.1, 0.12,  0.1,  0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.14, 0.12], dtype = float)
    cut_arr[:,1] = np.array([0.12,  0.1,  0.1,  0.1, 0.12, 0.12, 0.12, 0.12, 0.14, 0.12, 0.16, 0.12, 0.12, 0.12, 0.14, 0.12], dtype = float)
    cut_arr[:,2] = np.array([ 0.1, 0.08, 0.08, 1000,  0.1,  0.1, 0.08, 1000, 0.12,  0.1,  0.1, 1000,  0.1, 0.12, 0.12, 1000], dtype = float)
    cut_arr[:,3] = np.array([ 0.1, 0.08, 0.08, 1000,  0.1,  0.1, 0.08, 1000, 0.12,  0.1,  0.1, 1000,  0.1, 0.12, 0.12, 1000], dtype = float)
    cut_arr[:,4] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.12, 0.12, 1000, 0.12, 0.12, 0.14, 1000], dtype = float)
    cut_arr[:,5] = np.array([0.08, 0.06, 0.08, 0.06,  0.1, 0.08, 0.06, 0.08,  0.1, 0.08,  0.1, 0.08,  0.1,  0.1,  0.1, 0.08], dtype = float)
    cut_arr[:,6] = np.array([1000, 0.06,  0.1, 0.06, 1000,  0.1, 0.06, 0.08, 1000,  0.1,  0.1, 0.08, 1000,  0.1,  0.1, 0.08], dtype = float)
    """
    cut_arr = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr[:,0] = np.array([0.06, 0.04, 0.06, 0.06, 0.06, 0.08, 0.06,  0.1, 0.08, 0.06, 0.08, 0.06, 0.08,  0.1, 0.08, 0.08], dtype = float)
    cut_arr[:,1] = np.array([ 0.1, 0.04, 0.06, 0.06, 0.08, 0.08,  0.1,  0.1, 0.08, 0.06,  0.1, 0.06, 0.08, 0.12, 0.08, 0.08], dtype = float)
    cut_arr[:,2] = np.array([0.06, 0.04, 0.04, 1000, 0.06, 0.06, 0.06, 1000, 0.08, 0.06, 0.08, 1000, 0.06, 0.08, 0.08, 1000], dtype = float)
    cut_arr[:,3] = np.array([0.06, 0.04, 0.04, 1000, 0.06, 0.06, 0.06, 1000, 0.08, 0.06, 0.08, 1000, 0.06, 0.08, 0.08, 1000], dtype = float)
    cut_arr[:,4] = np.array([0.06, 0.04, 0.06, 1000, 0.06, 0.08, 0.08, 1000, 0.08, 0.06, 0.08, 1000, 0.08,  0.1, 0.08, 1000], dtype = float)
    cut_arr[:,5] = np.array([0.06, 0.04, 0.06,  0.2, 0.06, 0.06, 0.04, 0.06,  0.1, 0.04, 0.06, 0.06, 0.06, 0.08, 0.06, 0.06], dtype = float)
    cut_arr[:,6] = np.array([1000, 0.04, 0.06, 0.16, 1000, 0.06, 0.04, 0.16, 1000, 0.06, 0.06, 0.14, 1000, 0.08, 0.08, 0.16], dtype = float)

    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.06, 0.04], dtype = float)
    cut_arr_cut[:,1] = np.array([0.08, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04], dtype = float)
    cut_arr_cut[:,2] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.06, 0.04, 0.06, 1000, 0.06, 0.05, 0.06, 1000, 0.06, 0.06, 0.06, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([0.04, 0.04, 0.04,  0.1, 0.04, 0.04, 0.12, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04], dtype = float)
    cut_arr_cut[:,6] = np.array([1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.12, 0.02, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.08, 0.06], dtype = float)
    """
    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([0.16, 0.12, 0.12, 0.12, 0.16, 0.12, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16, 0.16, 0.14], dtype = float)
    cut_arr_cut[:,1] = np.array([0.16, 0.12, 0.12, 0.14, 0.16, 0.14, 0.14, 0.16, 0.16, 0.14,  0.2, 0.14, 0.14, 0.14, 0.16, 0.16], dtype = float)
    cut_arr_cut[:,2] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.12, 0.14, 1000, 0.12, 0.14, 0.16, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.14, 0.14, 1000, 0.14, 0.14, 0.14, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.14, 0.12, 0.12, 1000, 0.16, 0.16, 0.12, 1000, 0.18, 0.14, 0.16, 1000, 0.16, 0.16, 0.16, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([ 0.1, 0.08, 0.12, 0.08,  0.1, 0.12, 0.08, 0.12, 0.12,  0.1, 0.12, 0.12,  0.1, 0.14, 0.14, 0.12], dtype = float)
    cut_arr_cut[:,6] = np.array([1000, 0.06,  0.1, 0.08, 1000,  0.1, 0.06, 0.08, 1000, 0.12, 0.14,  0.1, 1000, 0.14, 0.12,  0.1], dtype = float)
    """
    cut_arr_cut = np.full((16, g_dim), np.nan, dtype = float)
    cut_arr_cut[:,0] = np.array([ 0.1, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.12,  0.1, 0.08, 0.16, 0.06,  0.1, 0.14,  0.1,  0.1], dtype = float)
    cut_arr_cut[:,1] = np.array([0.14, 0.06, 0.06, 0.06,  0.1,  0.1, 0.14, 0.12, 0.12, 0.08, 0.16, 0.08,  0.1, 0.14, 0.12,  0.1], dtype = float)
    cut_arr_cut[:,2] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000,  0.1,  0.1,  0.1, 1000], dtype = float)
    cut_arr_cut[:,3] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000, 0.08,  0.1, 0.08, 1000], dtype = float)
    cut_arr_cut[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.08,  0.1, 0.08, 1000,  0.1, 0.08,  0.1, 1000,  0.1, 0.12,  0.1, 1000], dtype = float)
    cut_arr_cut[:,5] = np.array([0.06, 0.04, 0.06, 0.04, 0.08, 0.08, 0.04,  0.1, 0.12, 0.06, 0.08, 0.06, 0.08,  0.1,  0.1, 0.08], dtype = float)
    cut_arr_cut[:,6] = np.array([1000, 0.04, 0.06,  0.2, 1000, 0.06, 0.04, 0.18, 1000, 0.06, 0.08, 0.18, 1000,  0.1, 0.08, 0.18], dtype = float)


min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16, g_dim), 0, dtype = float)
unix_ratio_rf_cut_map_good = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_good_kind = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_good_final = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_bad = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_bad_kind = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_bad_final = np.copy(unix_ratio_rf_cut_map)

md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013= int(md_2013_r.timestamp())
md_2020 = datetime(2020, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020= int(md_2020_r.timestamp())

unix_init = np.copy(unix_2013)
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)

ratio_map = np.full((len(unix_min_bins[:-1]), 16), 0, dtype = float)

cw_h5_path = '/misc/disk19/users/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
unix_txt = hf['bad_unix_time'][:]
txt_map = np.histogram(unix_txt, bins = unix_min_bins)[0].astype(int)
del hf

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')

    unix_time = hf['unix_min_bins'][:-1]
    unix_idx = (unix_time - unix_init)//60
    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day

    bad_idx = np.in1d(unix_time, unix_txt)
    good_idx = ~bad_idx
    #bad_idx_len = np.count_nonzero(bad_idx)
    #if bad_idx_len != 0:
    #    print(d_run_tot[r], bad_idx_len)

    u_r_map = hf['unix_ratio_rf_cut_map'][:]
    u_r_map_good = np.copy(u_r_map)
    u_r_map_good[bad_idx,:,:] = 0
    u_r_map_bad = np.copy(u_r_map)
    u_r_map_bad[good_idx,:,:] = 0

    unix_ratio_rf_cut_map[min_idx,:,:,g_idx] += u_r_map
    unix_ratio_rf_cut_map_good[min_idx,:,:,g_idx] += u_r_map_good
    unix_ratio_rf_cut_map_bad[min_idx,:,:,g_idx] += u_r_map_bad

    ratio_map[unix_idx] = hf['unix_ratio_rf_cut_map_max'][:]

      
    #if Station == 3:
    #    continue

    u_r_map_y = np.copy(u_r_map)
    u_r_map_y = (u_r_map_y > 0.0001).astype(int)
    u_r_map_y = u_r_map_y.astype(float)
    u_r_map_y *= ratio_bin_center[np.newaxis, :, np.newaxis]
    u_r_map_y = np.nanmax(u_r_map_y, axis = 1)

    u_r_map_y_cut = (u_r_map_y > cut_arr[:, g_idx][np.newaxis, :]).astype(int)
    u_r_map_y_cut = np.nansum(u_r_map_y_cut, axis = 1)
    u_r_map_good_kind = np.copy(u_r_map)
    u_r_map_good_kind[u_r_map_y_cut > 1,:,:] = 0
    u_r_map_bad_kind = np.copy(u_r_map)
    u_r_map_bad_kind[u_r_map_y_cut < 2,:,:] = 0
    unix_ratio_rf_cut_map_good_kind[min_idx,:,:,g_idx] += u_r_map_good_kind
    unix_ratio_rf_cut_map_bad_kind[min_idx,:,:,g_idx] += u_r_map_bad_kind
    
    #if Station == 3:
    #    continue
     
    u_r_map_y_cut = (u_r_map_y > cut_arr_cut[:, g_idx][np.newaxis, :]).astype(int)
    u_r_map_y_cut = np.nansum(u_r_map_y_cut, axis = 1)
    u_r_map_good_kind = np.copy(u_r_map)
    u_r_map_good_kind[u_r_map_y_cut > 1,:,:] = 0
    u_r_map_bad_kind = np.copy(u_r_map)
    u_r_map_bad_kind[u_r_map_y_cut < 2,:,:] = 0
    unix_ratio_rf_cut_map_good_final[min_idx,:,:,g_idx] += u_r_map_good_kind
    unix_ratio_rf_cut_map_bad_final[min_idx,:,:,g_idx] += u_r_map_bad_kind
    
    del hf

days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
txt_map = np.reshape(txt_map, (days_len, mins_len))
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Log_Full_{d_type}_A{Station}.h5'
#file_name = f'CW_Log_{d_type}_A{Station}_v9.5.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('txt_map', data=txt_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_good', data=unix_ratio_rf_cut_map_good, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_good_kind', data=unix_ratio_rf_cut_map_good_kind, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_good_final', data=unix_ratio_rf_cut_map_good_final, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_bad', data=unix_ratio_rf_cut_map_bad, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_bad_kind', data=unix_ratio_rf_cut_map_bad_kind, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_bad_final', data=unix_ratio_rf_cut_map_bad_final, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






