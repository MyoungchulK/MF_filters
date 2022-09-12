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
ri_key = 'run'
ri_key_len = len(ri_key)
rf_key = '.h5'

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_mf_sim/'
d_list, d_run_tot, d_run_range = file_sorter(d_path + '*noise*')
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf_sim/'

mf_bins = np.linspace(0,10,50+1)
mf_bin_center = (mf_bins[1:] + mf_bins[:-1]) / 2
mf_bin_len = len(mf_bin_center)
reco_bins = np.linspace(0,1,50+1)
reco_bin_center = (reco_bins[1:] + reco_bins[:-1]) / 2
reco_bin_len = len(reco_bin_center)

mf_noise_hist = np.full((2, config_len, mf_bin_len, reco_bin_len), 0, dtype = float)
mf_signal_hist = np.full((2, 3, config_len, mf_bin_len, reco_bin_len), 0, dtype = float)

pnu_bins = np.linspace(7,13,60+1)
pnu_bin_center = (pnu_bins[1:] + pnu_bins[:-1]) / 2
pnu_bin_len = len(pnu_bin_center)
reco_pnu_hist = np.full((2, 3, config_len, pnu_bin_len, reco_bin_len), 0, dtype = float)
mf_pnu_hist = np.full((2, 3, config_len, pnu_bin_len, mf_bin_len), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1

    ri_idx = file_name.find(ri_key)
    rf_idx = file_name.find(rf_key, ri_idx + ri_key_len)
    r_idx = int(file_name[ri_idx + ri_key_len:rf_idx])

    n_path = f'{m_path}mf_AraOut.A{Station}_C{c_idx+1}_E10000_noise_rayl.txt.run{r_idx}.h5'
    hf = h5py.File(n_path, 'r')
    evt_wise = hf['evt_wise'][:]
    del hf

    hf = h5py.File(d_list[r], 'r')
    coef = hf['coef'][:,1]

    mf_noise_hist[0, c_idx] += np.histogram2d(evt_wise[0], coef[0], bins = (mf_bins, reco_bins))[0] 
    mf_noise_hist[1, c_idx] += np.histogram2d(evt_wise[1], coef[1], bins = (mf_bins, reco_bins))[0] 
    
    del hf, evt_wise, file_name, i_idx, f_idx, c_idx

d_list, d_run_tot, d_run_range = file_sorter(d_path+'*signal*')
w_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/Weight_Sim_A{Station}.h5'
hf_w = h5py.File(w_path, 'r')
evt_rate = hf_w['evt_rate_livetime'][:]
del hf_w
evt_rate_pass = np.copy(evt_rate)

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

    ri_idx = file_name.find(ri_key)
    rf_idx = file_name.find(rf_key, ri_idx + ri_key_len)
    r_idx = int(file_name[ri_idx + ri_key_len:rf_idx])

    n_path = f'{m_path}mf_AraOut.A{Station}_C{c_idx+1}_E100_Nu{fla}_signal_rayl.txt.run{r_idx}.h5'
    hf = h5py.File(n_path, 'r')
    evt_wise = hf['evt_wise'][:]
    del hf

    hf = h5py.File(d_list[r], 'r')
    coef = hf['coef'][:,1]
    pnu = np.log10(hf['pnu'][:]/1e9)
    #print(pnu)

    pass_idx = np.logical_and(evt_wise[0] > 0.205, coef[0] > 0.175)
    evt_rate_pass[fla_idx, c_idx, r_idx, ~pass_idx] = np.nan

    mf_signal_hist[0, fla_idx, c_idx] += np.histogram2d(evt_wise[0], coef[0], bins = (mf_bins, reco_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0] 
    mf_signal_hist[1, fla_idx, c_idx] += np.histogram2d(evt_wise[1], coef[1], bins = (mf_bins, reco_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0] 
    
    reco_pnu_hist[0, fla_idx, c_idx] += np.histogram2d(pnu, coef[0], bins = (pnu_bins, reco_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0]
    reco_pnu_hist[1, fla_idx, c_idx] += np.histogram2d(pnu, coef[1], bins = (pnu_bins, reco_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0]

    mf_pnu_hist[0, fla_idx, c_idx] += np.histogram2d(pnu, evt_wise[0], bins = (pnu_bins, mf_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0]
    mf_pnu_hist[1, fla_idx, c_idx] += np.histogram2d(pnu, evt_wise[1], bins = (pnu_bins, mf_bins), weights = evt_rate[fla_idx, c_idx, r_idx])[0]

    del hf, evt_wise, file_name, i_idx, f_idx, c_idx, fi_idx, ff_idx, fla, fla_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Reco_MF_Map_W_Sim_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('pnu_bins', data=pnu_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bins', data=mf_bins, compression="gzip", compression_opts=9)
hf.create_dataset('reco_bins', data=reco_bins, compression="gzip", compression_opts=9)
hf.create_dataset('pnu_bin_center', data=pnu_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bin_center', data=mf_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('reco_bin_center', data=reco_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_noise_hist', data=mf_noise_hist, compression="gzip", compression_opts=9)
hf.create_dataset('mf_signal_hist', data=mf_signal_hist, compression="gzip", compression_opts=9)
hf.create_dataset('reco_pnu_hist', data=reco_pnu_hist, compression="gzip", compression_opts=9)
hf.create_dataset('mf_pnu_hist', data=mf_pnu_hist, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pass', data=evt_rate_pass, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
