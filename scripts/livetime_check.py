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

q_name = ['bad block length', 'bad block index', 'block gap', 'bad dda index', 'bad channel mask',
                        'single block', 'rf win', 'cal win', 'soft win', 'first minute',
                        'dda voltage', 'bad cal min rate', 'bad cal sec rate', 'bad soft sec rate', 'no rf cal sec rate', 'bad l1 rate',
                        'short runs', 'bad unix time', 'bad run', 'cw log', 'cw ratio', 'empty slot!', 'unlock calpulser',
                        'zero ped', 'single ped', 'low ped', 'known bad ped', 'calpuler cut', 'surface cut', 'op antenna cut']
print(len(q_name))

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
years = np.copy(configs)
del bad_runs

live_tot = np.full((d_len, 3), 0, dtype = float)
live_indi = np.full((d_len, len(q_name), 3), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    con = hf['config'][:]
    config = con[2]
    configs[r] = config
    years[r] = con[3]
    tot_live = np.nansum(hf['tot_qual_live_time'][:])
    bad_live = np.nansum(hf['tot_qual_sum_bad_live_time'][:])
    bad_indi_live = np.nansum(hf['tot_qual_bad_live_time'][:], axis = 0)
    live_tot[r, 0] = tot_live
    live_tot[r, 1] = tot_live - bad_live
    live_tot[r, 2] = bad_live
    live_indi[r, :, 0] = tot_live
    live_indi[r, :, 1] = tot_live - bad_indi_live
    live_indi[r, :, 2] = bad_indi_live
    del hf, con, tot_live, bad_live, bad_indi_live

summ = np.nansum(live_tot, axis = 0)
print(summ)
print(np.round((summ/summ[0])*100, 2))
summ_indi = np.nansum(live_indi, axis = 0)
for t in range(len(q_name)):
    print(f'{int(t + 1)}) {q_name[t]}:', np.round((summ_indi[t]/summ_indi[t, 0])*100, 2))
print(np.nansum(np.round((summ_indi/summ_indi[:, 0][:, np.newaxis])*100, 2), axis = 0))

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_Check_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('years', data=years, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('live_tot', data=live_tot, compression="gzip", compression_opts=9)
hf.create_dataset('live_indi', data=live_indi, compression="gzip", compression_opts=9)
hf.create_dataset('summ', data=summ, compression="gzip", compression_opts=9)
hf.create_dataset('summ_indi', data=summ_indi, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






