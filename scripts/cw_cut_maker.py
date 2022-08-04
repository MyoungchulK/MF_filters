import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
num_ants = 16

if Station == 2:
            num_configs = 6
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 1000], dtype = float)
            cw_arr_04[:,1] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)

            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.16, 0.16, 0.16, 0.14, 0.14, 0.14, 0.14, 0.18,  0.2, 0.16, 0.24, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.16, 0.16,  0.2, 0.14, 0.12, 0.24, 0.14,  0.2,  0.2, 0.18, 0.24, 0.16, 0.18, 0.16, 1000], dtype = float)
            cw_arr_025[:,2] = np.array([0.14, 0.16, 0.14, 0.14,  0.1,  0.1, 0.14, 0.14, 0.18, 0.18, 0.18, 0.26, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.16, 0.26, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.14, 0.18, 0.14, 0.12, 0.14, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([0.12, 0.12,  0.1, 0.12,  0.1,  0.1,  0.1,  0.1, 0.14, 0.16, 0.14, 0.18, 0.12, 0.12, 0.12, 1000], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([0.06, 0.18, 0.06, 0.14, 0.12, 0.08, 0.08, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08,  0.2,  0.1, 1000], dtype = float)
            cw_arr_0125[:,1] = np.array([0.06,  0.2, 0.06, 0.18,  0.2, 0.08,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.16, 0.06, 0.12, 0.14,  0.1,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.14, 0.04, 0.08, 0.22, 0.08, 0.06, 0.06,  0.1,  0.1, 0.08, 0.08, 0.06, 0.14, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.04, 0.14, 0.04, 0.06, 0.22, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.06, 0.06, 0.14, 0.06, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.04, 0.08, 0.04, 0.06, 0.14, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.08, 0.06, 0.12, 0.08, 1000], dtype = float)

if Station == 3:
            num_configs = 7
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.06, 0.04], dtype = float)
            cw_arr_04[:,1] = np.array([0.08, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.06, 0.04, 0.06, 1000, 0.06, 0.05, 0.06, 1000, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04,  0.1, 0.04, 0.04, 0.12, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04], dtype = float)
            cw_arr_04[:,6] = np.array([1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.12, 0.02, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.08, 0.06], dtype = float)

            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.12, 0.12, 0.12, 0.16, 0.12, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16, 0.16, 0.14], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.12, 0.12, 0.14, 0.16, 0.14, 0.14, 0.16, 0.16, 0.14,  0.2, 0.14, 0.14, 0.14, 0.16, 0.16], dtype = float)
            cw_arr_025[:,2] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.12, 0.14, 1000, 0.12, 0.14, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.14, 0.14, 1000, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.14, 0.12, 0.12, 1000, 0.16, 0.16, 0.12, 1000, 0.18, 0.14, 0.16, 1000, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([ 0.1, 0.08, 0.12, 0.08,  0.1, 0.12, 0.08, 0.12, 0.12,  0.1, 0.12, 0.12,  0.1, 0.14, 0.14, 0.12], dtype = float)
            cw_arr_025[:,6] = np.array([1000, 0.06,  0.1, 0.08, 1000,  0.1, 0.06, 0.08, 1000, 0.12, 0.14,  0.1, 1000, 0.14, 0.12,  0.1], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([ 0.1, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.12,  0.1, 0.08, 0.16, 0.06,  0.1, 0.14,  0.1,  0.1], dtype = float)
            cw_arr_0125[:,1] = np.array([0.14, 0.06, 0.06, 0.06,  0.1,  0.1, 0.14, 0.12, 0.12, 0.08, 0.16, 0.08,  0.1, 0.14, 0.12,  0.1], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000,  0.1,  0.1,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000, 0.08,  0.1, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.08,  0.1, 0.08, 1000,  0.1, 0.08,  0.1, 1000,  0.1, 0.12,  0.1, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.06, 0.04, 0.06, 0.04, 0.08, 0.08, 0.04,  0.1, 0.12, 0.06, 0.08, 0.06, 0.08,  0.1,  0.1, 0.08], dtype = float)
            cw_arr_0125[:,6] = np.array([1000, 0.04, 0.06,  0.2, 1000, 0.06, 0.04, 0.18, 1000, 0.06, 0.08, 0.18, 1000,  0.1, 0.08, 0.18], dtype = float)

# sort
d_path_0125 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
d_list_0125, d_run_tot_0125, d_run_range = file_sorter(d_path_0125)
d_path_025 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
d_list_025, d_run_tot_025, d_run_range = file_sorter(d_path_025)
d_path_04 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_04/*'
d_list_04, d_run_tot, d_run_range = file_sorter(d_path_04)
del d_path_0125, d_path_025, d_path_04, d_run_range
cut_label = ['cw_cut_04', 'cw_cut_025', 'cw_cut_0125']

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/'
if not os.path.exists(r_path):
    os.makedirs(r_path)

for r in range(len(d_run_tot)):
    
    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    cut_tot = np.full((16, 3), np.nan, dtype = float)
    cut_tot[:,0] = cw_arr_04[:,g_idx]
    cut_tot[:,1] = cw_arr_025[:,g_idx]
    cut_tot[:,2] = cw_arr_0125[:,g_idx]

    file_name = f'{r_path}cw_cut_A{Station}_R{d_run_tot[r]}.h5'
    hf_r = h5py.File(file_name, 'w')
    print(file_name) 

    run_list = [d_list_04[r], d_list_025[r], d_list_0125[r]]
    for h in range(3):
        hf = h5py.File(run_list[h], 'r')
        ratio = hf['sub_ratio'][:]                      # (pad,ant,evt)
        ratio_max = np.nanmax(ratio, axis = 0)          # (ant,evt) max from pad
        r_flag = ratio_max > cut_tot[:,h][:, np.newaxis]# (ant,evt) bool for each ent
        r_count = np.count_nonzero(r_flag, axis = 0)    # (evt)     count true from ant
        del ratio, ratio_max, r_flag

        evt_num = hf['evt_num'][:]
        if h == 0:
            hf_r.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
        clean_evt = hf['clean_evt'][:]
        evt_idx = np.in1d(evt_num, clean_evt)
        del hf, evt_num, clean_evt

        cw_cut_r = np.full((16, len(evt_num)), 0, dtype = int)
        for a in range(16):
            cw_cut_r[a, evt_idx] = (r_count > a).astype(int)
        del r_count, evt_idx
        hf_r.create_dataset(cut_label[h], data=cw_cut_r, compression="gzip", compression_opts=9)
    hf_r.close()
    del run_list, cut_tot, g_idx, file_name


print('Done!')





