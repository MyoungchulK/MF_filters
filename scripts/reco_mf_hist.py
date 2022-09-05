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
if Station == 2:
    config_len = 6
if Station == 3:
    config_len = 7

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_mf/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

mf_bins = np.linspace(0,15,1500+1)
mf_bin_center = (mf_bins[1:] + mf_bins[:-1]) / 2
mf_bin_len = len(mf_bin_center)
mf_hist = np.full((2, mf_bin_len, config_len), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    c_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    evt_wise = hf['coef'][:,1]
    del hf

    mf_hist[0, :, c_idx] += np.histogram(evt_wise[0], bins = mf_bins)[0] 
    mf_hist[1, :, c_idx] += np.histogram(evt_wise[1], bins = mf_bins)[0]
    del evt_wise, c_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Reco_MF_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('mf_bins', data=mf_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bin_center', data=mf_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_hist', data=mf_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
