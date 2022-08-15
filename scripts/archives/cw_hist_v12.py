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
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])

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

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path_0125 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
d_list_0125, d_run_tot_0125, d_run_range = file_sorter(d_path_0125)
d_path_025 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
d_list_025, d_run_tot_025, d_run_range = file_sorter(d_path_025)
d_path_04 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_04/*'
d_list_04, d_run_tot, d_run_range = file_sorter(d_path_04)
del d_path_0125, d_path_025, d_path_04, d_run_range

#mwx
cw_h5_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/radiosonde_data/radius_tot/'
mwx_name = f'{cw_h5_path}A{Station}_mwx_R.h5'
hf = h5py.File(mwx_name, 'r')
cw_unix_time = hf['cw_unix_time'][:]
del hf, cw_h5_path, mwx_name

ratio_bins = np.linspace(0,1,50+1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2

cw_hist = np.full((3, len(ratio_bin_center), num_ants, num_configs), 0, dtype = float)
cw_hist_good = np.full((3, 2, len(ratio_bin_center), num_ants, num_configs), 0, dtype = float)
cw_hist_bad = np.copy(cw_hist_good)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_f:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    cut_tot = np.full((16, 3), np.nan, dtype = float)
    cut_tot[:,0] = cw_arr_04[:,g_idx]
    cut_tot[:,1] = cw_arr_025[:,g_idx]
    cut_tot[:,2] = cw_arr_0125[:,g_idx]

    run_list = [d_list_04[r], d_list_025[r], d_list_0125[r]]
    for h in range(3):
        hf = h5py.File(run_list[h], 'r')
        weight = hf['sub_weight'][:]
        ratio = hf['sub_ratio'][:]
        ratio_max = np.nanmax(ratio, axis = 0)
        r_flag = ratio_max > cut_tot[:,h][:, np.newaxis]
        r_count = np.count_nonzero(r_flag, axis = 0)
        flag_1_ant = r_count > 0
        flag_2_ant = r_count > 1
        del ratio_max, r_flag, r_count

        if h == 0:
            if g_idx > 4:
                unix_time = hf['clean_unix'][:]
                mwx_evt_idx = np.in1d(unix_time, cw_unix_time)
                del unix_time 
            else:
                mwx_evt_idx = np.full((len(ratio[0,0,:])), False, dtype = bool)
        flag_1_ant_tot = np.logical_or(flag_1_ant, mwx_evt_idx)
        flag_2_ant_tot = np.logical_or(flag_2_ant, mwx_evt_idx)
        del hf, flag_1_ant, flag_2_ant
        if h == 2: del mwx_evt_idx

        for ant in range(num_ants):
            cw_hist[h, :, ant, g_idx] += np.histogram(ratio[:, ant].flatten(), bins = ratio_bins, weights = weight[:, ant].flatten())[0]
            cw_hist_good[h, 0, :, ant, g_idx] += np.histogram(ratio[:, ant, ~flag_1_ant_tot].flatten(), bins = ratio_bins, weights = weight[:, ant, ~flag_1_ant_tot].flatten())[0]
            cw_hist_good[h, 1, :, ant, g_idx] += np.histogram(ratio[:, ant, ~flag_2_ant_tot].flatten(), bins = ratio_bins, weights = weight[:, ant, ~flag_2_ant_tot].flatten())[0]
            cw_hist_bad[h, 0, :, ant, g_idx] += np.histogram(ratio[:, ant, flag_1_ant_tot].flatten(), bins = ratio_bins, weights = weight[:, ant, flag_1_ant_tot].flatten())[0]
            cw_hist_bad[h, 1, :, ant, g_idx] += np.histogram(ratio[:, ant, flag_2_ant_tot].flatten(), bins = ratio_bins, weights = weight[:, ant, flag_2_ant_tot].flatten())[0]
        del weight, ratio, flag_1_ant_tot, flag_2_ant_tot 
    del run_list, cut_tot, g_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Result_A{Station}_{count_i}_{count_f}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('cw_hist', data=cw_hist, compression="gzip", compression_opts=9)
hf.create_dataset('cw_hist_good', data=cw_hist_good, compression="gzip", compression_opts=9)
hf.create_dataset('cw_hist_bad', data=cw_hist_bad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






