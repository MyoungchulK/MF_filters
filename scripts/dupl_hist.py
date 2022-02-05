import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import file_sorter
from tools.utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dupl/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_all = []
run_arr = []
run_arr_all = []
dupl = []
dupl_rf = []
dupl_rf_w_cut = []
num_evts = []
rf_evts = []
pre_cut = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    trig_type = hf['trig_type'][:]
    qual_cut = hf['pre_qual_cut'][:]

    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])

    num_evts_run = len(hf['evt_num'][:])
    num_evts.append(num_evts_run)
    rf_evt = len(hf['clean_rf_evt'][:])
    rf_evts.append(rf_evt)

    qual_cut_count = np.count_nonzero(qual_cut, axis = 0)
    pre_cut.append(qual_cut_count)

    dupl.append(hf['dupl_hist'][:])
    dupl_rf.append(hf['dupl_rf_hist'][:])

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    dupl_rf_w_cut.append(hf['dupl_rf_hist_w_cut'][:])

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Dupl_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=np.asarray(num_evts), compression="gzip", compression_opts=9)
hf.create_dataset('rf_evts', data=np.asarray(rf_evts), compression="gzip", compression_opts=9)
hf.create_dataset('pre_cut', data=np.asarray(pre_cut), compression="gzip", compression_opts=9)
hf.create_dataset('dupl', data=np.asarray(dupl), compression="gzip", compression_opts=9)
hf.create_dataset('dupl_rf', data=np.asarray(dupl_rf), compression="gzip", compression_opts=9)
hf.create_dataset('dupl_rf_w_cut', data=np.asarray(dupl_rf_w_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)









