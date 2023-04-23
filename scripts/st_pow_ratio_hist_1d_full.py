import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
#from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_full/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range, d_path 

r_bins = np.linspace(0, 30, 300 + 1)
r_bin_center = (r_bins[1:] + r_bins[:-1]) / 2
r_bin_len = len(r_bin_center)

pow_r = np.full((d_len, r_bin_len, 3), 0, dtype = int)
pow_r_cut = np.copy(pow_r)
del r_bin_len

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
del bad_runs

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs[r] = hf['config'][2]
    evt = hf['evt_num'][:]
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    rms = hf['rms'][:]
    del hf, trig

    bad_ant = known_issue.get_bad_antenna(d_run_tot[r])
    rms[bad_ant] = np.nan
    pow_n = rms ** 2
    del rms, bad_ant

    pow_n_avg = np.full((4, len(evt)), np.nan, dtype = float)
    for m in range(4):
        pow_n_avg[m] = np.nanmean(pow_n[m::4], axis = 0)
    del pow_n

    pow_n_avg_sort = -np.sort(-pow_n_avg, axis = 0)    
    del pow_n_avg

    pow_ratio = pow_n_avg_sort[0] / pow_n_avg_sort[1]
    del pow_n_avg_sort

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    #evt_full = hf_q['evt_num'][:]
    cut = hf_q['tot_qual_cut_sum'][:] != 0
    #cut = np.in1d(evt, evt_full[qual])
    del q_name, hf_q, evt
    #del q_name, hf_q, qual, evt_full, evt

    pow_ratio_cut = np.copy(pow_ratio)
    pow_ratio_cut[cut] = np.nan
    del cut

    for t in range(3):
        pow_r[r, :, t] = np.histogram(pow_ratio[t_list[t]], bins = r_bins)[0].astype(int)    
        if b_runs[r]: continue
        pow_r_cut[r, :, t] = np.histogram(pow_ratio_cut[t_list[t]], bins = r_bins)[0].astype(int)    
    del pow_ratio_cut, pow_ratio, t_list, rf_t, cal_t, soft_t

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Pow_Ratio_1d_full_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r', data=pow_r, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r_cut', data=pow_r_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






