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

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/daq_cut/'
d_len = len(d_run_tot)

snr_bins = np.linspace(0,150,150+1)
snr_bin_center = (snr_bins[1:] + snr_bins[:-1]) / 2
snr_bin_len = len(snr_bin_center)

config_arr = np.full((d_len), 0, dtype = int)
run_arr = np.copy(d_run_tot)
snr_tot = np.full((snr_bin_len, 16, d_len), 0, dtype = int)
snr_rf = np.copy(snr_tot)
snr_cal = np.copy(snr_tot)
snr_soft = np.copy(snr_tot)
snr_rf_cut = np.copy(snr_tot)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr[r] = config

    trig = hf['trig_type'][:]
    snr_r = hf['snr'][:]
    snr_rf_r = snr_r[:, trig == 0]
    snr_cal_r = snr_r[:, trig == 1]
    snr_soft_r = snr_r[:, trig == 2]
    
    for a in range(16):
        snr_tot[:,a,r] = np.histogram(snr_r[a], bins = snr_bins)[0].astype(int)
        snr_rf[:,a,r] = np.histogram(snr_rf_r[a], bins = snr_bins)[0].astype(int)
        snr_cal[:,a,r] = np.histogram(snr_cal_r[a], bins = snr_bins)[0].astype(int)
        snr_soft[:,a,r] = np.histogram(snr_soft_r[a], bins = snr_bins)[0].astype(int)
    del snr_rf_r, snr_cal_r, snr_soft_r

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf_q = h5py.File(f'{q_path}daq_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cuts = hf_q['pre_qual_cut_sum'][:]
    del hf_q

    snr_rf_clean = snr_r[:, (trig == 0) & (cuts == 0)]
    for a in range(16):
        snr_rf_cut[:,a,r] = np.histogram(snr_rf_clean[a], bins = snr_bins)[0].astype(int)
    del hf, trig, snr_r, snr_rf_clean, cuts


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'SNR_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('snr_bins', data=snr_bins, compression="gzip", compression_opts=9)
hf.create_dataset('snr_bin_center', data=snr_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('snr_tot', data=snr_tot, compression="gzip", compression_opts=9)
hf.create_dataset('snr_rf', data=snr_rf, compression="gzip", compression_opts=9)
hf.create_dataset('snr_cal', data=snr_cal, compression="gzip", compression_opts=9)
hf.create_dataset('snr_soft', data=snr_soft, compression="gzip", compression_opts=9)
hf.create_dataset('snr_rf_cut', data=snr_rf_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






