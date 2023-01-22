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
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2:num_configs = 7
if Station == 3:num_configs = 9
num_ants = 16
num_trigs = 3

known_issue = known_issue_loader(Station, verbose = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l2/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

ratio_bins = np.linspace(-2, 2, 400 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_len = len(ratio_bin_center)

#output
ratio_dir = np.full((ratio_len, num_ants, num_trigs, 3, num_configs), 0, dtype = int)

count_ff = count_i + count_f

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    bad_ant = known_issue.get_bad_antenna(d_run_tot[r])

    hf = h5py.File(d_list[r], 'r')
    cw_ratio = 1 - hf['cw_ratio'][:]
    cw_ratio[bad_ant] = np.nan
    cw_ratio = np.sort(cw_ratio, axis = 0)
    trig_type = hf['trig_type'][:]
    evt_num = hf['evt_num'][:]
    del bad_ant

    q_hf = h5py.File(f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5', 'r')
    q_evt = q_hf['evt_num'][:]
    qual = q_hf['tot_qual_cut_sum'][:] != 0
    q_bad = np.in1d(evt_num, q_evt[qual])
    del q_hf, q_evt, qual

    for trig in range(num_trigs):
        trigs = trig_type == trig
        for q in range(3):
            if q == 0: bools = trigs
            elif q == 1: bools = np.logical_and(trigs, ~q_bad)
            else: bools = np.logical_and(trigs, q_bad)
            ratio_trig = cw_ratio[:, bools]
            for ant in range(num_ants):
                ratio_dir[:, ant, trig, q, g_idx] += np.histogram(ratio_trig[ant], bins = ratio_bins)[0].astype(int)
            del bools
        del trigs
    del g_idx, hf, cw_ratio, trig_type, q_bad

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Ratio_Direct_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_dir', data=ratio_dir, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






