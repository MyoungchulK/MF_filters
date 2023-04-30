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

c_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf/'

mf_bins = np.linspace(0,10,50+1)
mf_bin_center = (mf_bins[1:] + mf_bins[:-1]) / 2
mf_bin_len = len(mf_bin_center)
reco_bins = np.linspace(0,1,50+1)
reco_bin_center = (reco_bins[1:] + reco_bins[:-1]) / 2
reco_bin_len = len(reco_bin_center)

mf_hist = np.full((2, config_len, mf_bin_len, reco_bin_len), 0, dtype = float)

num_evts = np.full((config_len),0,dtype = float)
num_evts_pass = np.copy(num_evts)

for r in tqdm(range(len(d_run_tot))):

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    c_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    coef = hf['coef'][:,1]
    del hf

    hf_m = h5py.File(f'{m_path}mf_A{Station}_R{d_run_tot[r]}.h5', 'r')
    evt_wise = hf_m['evt_wise'][:]
    del hf_m

    hf_c = h5py.File(f'{c_path}cw_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cuts = hf_c['cw_qual_cut_sum'][:]
    del hf_c

    coef[:, cuts != 0] = np.nan
    evt_wise[:, cuts != 0] = np.nan

    coef[coef > 0.17] = np.nan
    evt_wise[evt_wise > 0.2] = np.nan
    
    num_evts[c_idx] += np.count_nonzero(np.logical_and(~np.isnan(coef[0]), ~np.isnan(evt_wise[0])))
    num_evts_pass[c_idx] += np.count_nonzero(np.logical_and(coef[0] > 0.175, evt_wise[0] > 0.205))

    mf_hist[0, c_idx] += np.histogram2d(evt_wise[0], coef[0], bins = (mf_bins, reco_bins))[0] 
    mf_hist[1, c_idx] += np.histogram2d(evt_wise[1], coef[1], bins = (mf_bins, reco_bins))[0]
    del evt_wise, c_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Reco_MF_Map_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('mf_bins', data=mf_bins, compression="gzip", compression_opts=9)
hf.create_dataset('reco_bins', data=reco_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bin_center', data=mf_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('reco_bin_center', data=reco_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_hist', data=mf_hist, compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('num_evts_pass', data=num_evts_pass, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
