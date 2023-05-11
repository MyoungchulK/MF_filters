import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
if Station == 2:
    config_len = 6
if Station == 3:
    config_len = 7

i_key = '_C'
i_key_len = len(i_key)
f_key = '_E1'
fi_key = '_Nu'
fi_key_len = len(fi_key)
ff_key = '_signal'

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_sim/'
d_list, d_run_tot, d_run_range = file_sorter(d_path + '*noise*')

mf_bins = np.linspace(0,15,1500+1)
mf_bin_center = (mf_bins[1:] + mf_bins[:-1]) / 2
mf_bin_len = len(mf_bin_center)
mf_noise_hist = np.full((2, mf_bin_len, config_len), 0, dtype = int)
mf_signal_hist = np.full((2, 3, mf_bin_len, config_len), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1

    hf = h5py.File(d_list[r], 'r')
    evt_wise = hf['coef'][:,1]

    mf_noise_hist[0, :, c_idx] += np.histogram(evt_wise[0], bins = mf_bins)[0] 
    mf_noise_hist[1, :, c_idx] += np.histogram(evt_wise[1], bins = mf_bins)[0] 
    
    del hf, evt_wise, file_name, i_idx, f_idx, c_idx

d_list, d_run_tot, d_run_range = file_sorter(d_path+'*signal*')

for r in tqdm(range(len(d_run_tot))):
    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1

    fi_idx = file_name.find(fi_key)
    ff_idx = file_name.find(ff_key, fi_idx + fi_key_len)
    fla = file_name[fi_idx + fi_key_len:ff_idx]
    if fla == 'E': fla_idx = 0
    if fla == 'Mu': fla_idx = 1
    if fla == 'Tau': fla_idx = 2

    hf = h5py.File(d_list[r], 'r')
    evt_wise = hf['coef'][:,1]

    mf_signal_hist[0, fla_idx, :, c_idx] += np.histogram(evt_wise[0], bins = mf_bins)[0] 
    mf_signal_hist[1, fla_idx, :, c_idx] += np.histogram(evt_wise[1], bins = mf_bins)[0] 
    
    del hf, evt_wise, file_name, i_idx, f_idx, c_idx, fi_idx, ff_idx, fla, fla_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Reco_Sim_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('mf_bins', data=mf_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bin_center', data=mf_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_noise_hist', data=mf_noise_hist, compression="gzip", compression_opts=9)
hf.create_dataset('mf_signal_hist', data=mf_signal_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
