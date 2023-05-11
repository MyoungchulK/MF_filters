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

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
print(bad_runs)
print(f'# of bad runs: {len(bad_runs)}')

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/freq_ratio/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)

r_bins = np.linspace(-1, 2, 3000 + 1, dtype = float)
r_bin_center = (r_bins[1:] + r_bins[:-1]) / 2
r_bin_len = len(r_bin_center)

r_hist = np.full((r_bin_len, 16, 3, num_configs), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    bad_idx = d_run_tot[r] in bad_runs
    if bad_idx:
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        continue

    evt = hf['evt_num'][:]
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    power = hf['power'][:]
    power_n = hf['power_notch'][:]
    ratio = 1 - power_n / power
    del hf, trig, power, power_n

    for a in range(16):
        for t in range(3):
            r_hist[:, a, t, g_idx] += np.histogram(ratio[a][t_list[t]], bins = (r_bins))[0].astype(int)
            if Station == 2 and g_idx == 0 and t == 0 and np.any(ratio[a][t_list[t]] > 0.19):
                print(Station, d_run_tot[r], a, g_idx, evt[t_list[t]][ratio[a][t_list[t]] > 0.19])

    del evt, ratio, rf_t, cal_t, soft_t, t_list 

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Freq_Ratio_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('r_hist', data=r_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






