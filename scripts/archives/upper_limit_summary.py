import numpy as np
import os, sys
import h5py
from tqdm import tqdm
import ROOT

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
num_pols = 2

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

hf_v = h5py.File(d_path+f'back_est_A{Station}_VPol.h5', 'r')
hf_h = h5py.File(d_path+f'back_est_A{Station}_HPol.h5', 'r')

s_ang = hf_v['s_ang'][:]
num_slos = len(s_ang)
bins_d = hf_v['bins_d'][:]
bin_center_d = hf_v['bin_center_d'][:]
bins_s = hf_v['bins_s'][:]
bin_center_s = hf_v['bin_center_s'][:]

map_d_bins_v = hf_v['map_d_bins'][:]
map_d_bin_center_v = hf_v['map_d_bin_center'][:]
map_d_bins_h = hf_h['map_d_bins'][:]
map_d_bin_center_h = hf_h['map_d_bin_center'][:]
map_d_len = len(map_d_bin_center_h[:, 0, 0])

back_median_v = hf_v['back_median'][:]
back_err_v = hf_v['back_err'][:]
back_median_h = hf_h['back_median'][:]
back_err_h = hf_h['back_err'][:]

#sig_int_v = hf_v['sig_int'][:]
#sig_eff_v = hf_v['sig_eff'][:]
#sig_int_h = hf_h['sig_int'][:]
#sig_eff_h = hf_h['sig_eff'][:]

num_trials = 10000
bin_width = 100
bin_edges_o = np.full((2, num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = float)
bin_edges_u = np.copy(bin_edges_o)
nobs_hist = np.full((bin_width, num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = int)
upl_hist = np.copy(nobs_hist)
upl_mean = np.full((num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = float)

fc = ROOT.TFeldmanCousins(0.90) 

for c in range(num_configs):
    for s in tqdm(range(num_slos)):
        for d in range(map_d_len):
            nobs = np.full((num_pols, num_trials), np.nan, dtype = float)
            upl = np.copy(nobs)
            for n in tqdm(range(num_trials)):
                nobs[0, n] = np.random.poisson(back_median_v[d, s, c])
                nobs[1, n] = np.random.poisson(back_median_h[d, s, c])
                upl[0, n] = fc.CalculateUpperLimit(nobs[0, n], back_median_v[d, s, c])
                upl[1, n] = fc.CalculateUpperLimit(nobs[1, n], back_median_h[d, s, c])
                print(upl[:, n])
            nobs_bins = np.linspace(np.nanmin(nobs, axis = 1), np.nanmax(nobs, axis = 1), bin_width + 1)
            bin_edges_o[0, :, d, s, c] = nobs_bins[0]
            bin_edges_o[1, :, d, s, c] = nobs_bins[-1]
            nobs_hist[:, 0, d, s, c] = np.histogram(nobs[0], bins = nobs_bins[:, 0])[0].astype(int)
            nobs_hist[:, 1, d, s, c] = np.histogram(nobs[1], bins = nobs_bins[:, 1])[1].astype(int)
            upl_bins = np.linspace(np.nanmin(upl, axis = 1), np.nanmax(upl, axis = 1), bin_width + 1)
            bin_edges_u[0, :, d, s, c] = upl_bins[0]
            bin_edges_u[1, :, d, s, c] = upl_bins[-1]
            upl_hist[:, 0, d, s, c] = np.histogram(upl[0], bins = upl_bins[:, 0])[0].astype(int)
            upl_hist[:, 1, d, s, c] = np.histogram(upl[1], bins = upl_bins[:, 1])[1].astype(int)
            upl_mean[:, d, s, c] = np.nanmean(upl, axis = 1)
            del nobs, upl, nobs_bins, upl_bins

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Upper_Limit_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






