import os, sys
import numpy as np
import h5py
from tqdm import tqdm

Station = int(sys.argv[1])

dpath = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
print(dpath)
rpath = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
print(rpath)
if not os.path.exists(rpath):
    os.makedirs(rpath)
os.chdir(rpath)

if Station == 2:
    num_configs = 7
if Station == 3:
    num_configs = 8
ant_c = 1
trig = np.array([0, 1, 2], dtype = int)
smear_len = np.array([1, 5, 10, 15, 20, 25], dtype = int)
run_arr = np.arange(0, 12000 + 1, 1000, dtype = int)
livetime = np.full((3, len(smear_len)), 0, dtype = int)
days_bins = np.linspace(0,2922, 2922 + 1, dtype = int)
hrs_bins = np.linspace(0, 1440, 1440 + 1, dtype = float) / 60

ratio_04_hist = np.full((50, 16, num_configs, len(trig), len(smear_len)), 0, dtype = int)
ratio_04_pass_hist = np.copy(ratio_04_hist)
ratio_04_cut_hist = np.copy(ratio_04_hist)
ratio_025_hist = np.copy(ratio_04_hist)
ratio_025_pass_hist = np.copy(ratio_04_hist)
ratio_025_cut_hist = np.copy(ratio_04_hist)
ratio_0125_hist = np.copy(ratio_04_hist)
ratio_0125_pass_hist = np.copy(ratio_04_hist)
ratio_0125_cut_hist = np.copy(ratio_04_hist)
print('hist done!')

ratio_04_map = np.full((2922, 1440, 16, len(trig), len(smear_len)), 0, dtype = int)
ratio_04_pass_map = np.copy(ratio_04_map)
ratio_04_cut_map = np.copy(ratio_04_map)
ratio_025_map = np.copy(ratio_04_map)
ratio_025_pass_map = np.copy(ratio_04_map)
ratio_025_cut_map = np.copy(ratio_04_map)
ratio_0125_map = np.copy(ratio_04_map)
ratio_0125_pass_map = np.copy(ratio_04_map)
ratio_0125_cut_map = np.copy(ratio_04_map)
print('map done!')

for t in range(len(trig)):
    for r in range(len(run_arr)):
        for s in range(len(smear_len)):
            d_name = f'{dpath}CW_Table_Test_Smear_Combine_Cut_full_v8_A{Station}_T{trig[t]}_C{ant_c}_S{smear_len[s]}_R{run_arr[r]}.h5'
            hf = h5py.File(d_name, 'r')
            print(d_name)           
 
            if t == 0 and r == 0 and s == 0:
                ratio_bins = hf['ratio_bins'][:]
                ratio_bin_center = hf['ratio_bin_center'][:]
                hrs_in_days = hf['hrs_in_days'][:]
                day_in_yrs = hf['day_in_yrs'][:]
            if t == 0:
                livetime[0, s] += np.nansum(hf['tot_sec'][:])
                livetime[1, s] += np.nansum(hf['tot_bad_sec_04'][:])
                livetime[2, s] += np.nansum(hf['tot_bad_sec_025'][:])
            ratio_04_hist[:, :, :, t, s] += hf['ratio_04_hist'][:]
            ratio_04_pass_hist[:, :, :, t, s] += hf['ratio_04_pass_hist'][:]
            ratio_04_cut_hist[:, :, :, t, s] += hf['ratio_04_cut_hist'][:]
            ratio_025_hist[:, :, :, t, s] += hf['ratio_025_hist'][:]
            ratio_025_pass_hist[:, :, :, t, s] += hf['ratio_025_pass_hist'][:]
            ratio_025_cut_hist[:, :, :, t, s] += hf['ratio_025_cut_hist'][:]
            ratio_0125_hist[:, :, :, t, s] += hf['ratio_0125_hist'][:]
            ratio_0125_pass_hist[:, :, :, t, s] += hf['ratio_0125_pass_hist'][:]
            ratio_0125_cut_hist[:, :, :, t, s] += hf['ratio_0125_cut_hist'][:]
            ratio_04_map[:, :, :, t, s] += hf['ratio_04_map'][:]
            ratio_04_pass_map[:, :, :, t, s] += hf['ratio_04_pass_map'][:]
            ratio_04_cut_map[:, :, :, t, s] += hf['ratio_04_cut_map'][:]
            ratio_025_map[:, :, :, t, s] += hf['ratio_025_map'][:]
            ratio_025_pass_map[:, :, :, t, s] += hf['ratio_025_pass_map'][:]
            ratio_025_cut_map[:, :, :, t, s] += hf['ratio_025_cut_map'][:]
            ratio_0125_map[:, :, :, t, s] += hf['ratio_0125_map'][:]
            ratio_0125_pass_map[:, :, :, t, s] += hf['ratio_0125_pass_map'][:]
            ratio_0125_cut_map[:, :, :, t, s] += hf['ratio_0125_cut_map'][:]
            del hf
            
print(livetime[1] /  livetime[0])
print(livetime[2] /  livetime[0])
       
file_name = f'{rpath}CW_Table_Test_Smear_Combine_Cut_full_tot_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('trig', data=trig, compression="gzip", compression_opts=9)
hf.create_dataset('smear_len', data=smear_len, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('days_bins', data=days_bins, compression="gzip", compression_opts=9)
hf.create_dataset('hrs_bins', data=hrs_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_hist', data=ratio_04_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_hist', data=ratio_04_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_hist', data=ratio_04_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_hist', data=ratio_025_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_hist', data=ratio_025_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_hist', data=ratio_025_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_hist', data=ratio_0125_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_hist', data=ratio_0125_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_hist', data=ratio_0125_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_map', data=ratio_04_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_map', data=ratio_04_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_map', data=ratio_04_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_map', data=ratio_025_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_map', data=ratio_025_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_map', data=ratio_025_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_map', data=ratio_0125_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_map', data=ratio_0125_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_map', data=ratio_0125_cut_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
