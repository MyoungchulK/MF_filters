import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
dd_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/'
#d_path = f'{dd_path}l2/*'
d_path = f'{dd_path}cw_ratio/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

q_path = f'{dd_path}qual_cut_full/'

#r_path = f'{dd_path}cw_ratio/'
#if not os.path.exists(r_path):    
#    os.makedirs(r_path)

num_ants = 16
num_trigs = 3
ratio_bins = np.linspace(-2, 2, 400 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_len = len(ratio_bin_center)
ant_bins = np.linspace(-0.5, 16.5, 17+1)
ant_bin_center = (ant_bins[1:] + ant_bins[:-1]) / 2

#known_issue = known_issue_loader(Station)

if Station == 2:num_configs = 7
if Station == 3:num_configs = 9
#ratio_map_tot = np.full((ratio_len, num_ants, num_trigs, num_configs), 0, dtype = int)
ratio_map_tot = np.full((ratio_len, num_ants+1, num_trigs, 3, num_configs), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
  
  #if r > 937:  
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    #bad_ant = known_issue.get_bad_antenna(d_run_tot[r])
    #good_ant = ~bad_ant
    #bad_ant = bad_ant.astype(int)

    hf = h5py.File(d_list[r], 'r')
    evt_num = hf['evt_num'][:]
    trig_type = hf['trig_type'][:]
    cw_ratio = hf['cw_ratio'][:]
    good_ant = hf['bad_ant'][:] == 0
    cw_ratio = cw_ratio[good_ant]
    del hf, good_ant

    q_hf = h5py.File(f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5', 'r')
    q_evt = q_hf['evt_num'][:]    
    qual = q_hf['tot_qual_cut_sum'][:] != 0
    q_bad = np.in1d(evt_num, q_evt[qual])
    del q_hf, q_evt, qual

    for t in range(num_trigs):
        for q in range(3):
            if q == 0: bools = trig_type == t
            elif q == 1: bools = np.logical_and(trig_type == t, ~q_bad)
            else: bools = np.logical_and(trig_type == t, q_bad)
            dat = cw_ratio[:, bools]
            for a in range(ratio_len):
                dat_c = np.count_nonzero(dat > ratio_bins[a], axis = 0)
                ratio_map_tot[a, :, t, q, g_idx] += np.histogram(dat_c, bins = ant_bins)[0].astype(int)
                del dat_c
            del bools, dat
    del q_bad

    """
    rf = cw_ratio[:, trig_type == 0][good_ant]
    cal = cw_ratio[:, trig_type == 1][good_ant]
    soft = cw_ratio[:, trig_type == 2][good_ant]
    del good_ant

    ratio_map = np.full((ratio_len, num_ants, num_trigs), 0, dtype = int)
    for a in range(ratio_len):
        rf_c = np.count_nonzero(rf > ratio_bins[a], axis = 0)
        ratio_map[a, :, 0] = np.histogram(rf_c, bins = ant_bins)[0].astype(int) 
        cal_c = np.count_nonzero(cal > ratio_bins[a], axis = 0)
        ratio_map[a, :, 1] = np.histogram(cal_c, bins = ant_bins)[0].astype(int)
        soft_c = np.count_nonzero(soft > ratio_bins[a], axis = 0)
        ratio_map[a, :, 2] = np.histogram(soft_c, bins = ant_bins)[0].astype(int)
        del rf_c, cal_c, soft_c
    del rf, cal, soft

    ratio_map_tot[:, :, :, g_idx] += ratio_map
    del g_idx
    
    hf_name = f'{r_path}cw_ratio_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(hf_name, 'w')
    hf.create_dataset('bad_ant', data=bad_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('cw_ratio', data=cw_ratio, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
    hf.create_dataset('ant_bins', data=ant_bins, compression="gzip", compression_opts=9)
    hf.create_dataset('ant_bin_center', data=ant_bin_center, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
    hf.close()
    del bad_ant, evt_num, trig_type, cw_ratio, hf_name, ratio_map
    """
    del g_idx, evt_num, trig_type, cw_ratio

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

#file_name = f'CW_Ratio_v1_A{Station}_R{count_i}.h5'
file_name = f'CW_Ratio_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map_tot', data=ratio_map_tot, compression="gzip", compression_opts=9)
hf.create_dataset('ant_bins', data=ant_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ant_bin_center', data=ant_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))

print('done!')




