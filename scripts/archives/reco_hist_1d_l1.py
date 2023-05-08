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

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)

map_r_z = np.full((d_len, z_bin_len, 2), 0, dtype = int) # v, h, trig
map_r_z_l1 = np.copy(map_r_z)
map_r_z_cut = np.copy(map_r_z)
map_r_z_cut_l1 = np.copy(map_r_z)
map_r_a = np.full((d_len, a_bin_len, 2), 0, dtype = int) # v, h, trig
map_r_a_l1 = np.copy(map_r_a)
map_r_a_cut = np.copy(map_r_a)
map_r_a_cut_l1 = np.copy(map_r_a)
map_r_c = np.full((d_len, c_bin_len, 2), 0, dtype = int) # v, h, trig
map_r_c_l1 = np.copy(map_r_c)
map_r_c_cut = np.copy(map_r_c)
map_r_c_cut_l1 = np.copy(map_r_c)
del z_bin_len, a_bin_len, c_bin_len

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
nan_counts = np.copy(configs)
del bad_runs

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs[r] = hf['config'][2]
    evt = hf['evt_num'][:]
    rf_t = hf['trig_type'][:] == 0
    rf_len = np.count_nonzero(rf_t)
    coord = hf['coord'][:, :, :, :, rf_t] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:, :, :, rf_t] # pol, rad, sol, evt
    coef_re = np.reshape(coef, (2, 4, -1))
    coef_max = np.nanmax(coef_re, axis = 1)
    #coef_max_idx = np.nanargmax(coef_re, axis = 1)
    coord_re = np.reshape(coord, (2, 2, 4, -1))
    coord_max = np.full((2, 2, rf_len), np.nan, dtype = float)
    counts = 0
    for rf in range(rf_len):
        try:
            coef_max_idx = np.nanargmax(coef_re[:, :, rf], axis = 1)
        except ValueError:
            counts += 1
            #print(coef_re[:, :, rf])
            continue
        coord_max[0, :, rf] = coord_re[0, :, coef_max_idx[0], rf]
        coord_max[1, :, rf] = coord_re[1, :, coef_max_idx[1], rf]
        del coef_max_idx
    nan_counts[r] = counts
    del hf, coord, coef, coef_re, coord_re, rf_len

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    cut = np.in1d(evt, evt_full[qual])[rf_t]
    qual_tot = hf_q['tot_qual_cut'][:]
    l1_cut_full = qual_tot[:, 15] != 0
    l1_cut = ~np.in1d(evt, evt_full[l1_cut_full])[rf_t]
    qual_tot[:, 14] = 0
    qual_tot[:, 15] = 0
    qual_tot = np.nansum(qual_tot, axis = 1)
    l1_only_cut_full = np.logical_and(l1_cut_full, qual_tot == 0) 
    l1_only_cut = ~np.in1d(evt, evt_full[l1_only_cut_full])[rf_t]
    del q_name, hf_q, evt, evt_full, qual, qual_tot, l1_cut_full, l1_only_cut_full, rf_t

    coef_clean = np.copy(coef_max)
    coef_clean[:, cut] = np.nan
    coef_l1_flag = np.copy(coef_max)
    coef_l1_flag[:, l1_cut] = np.nan
    coef_l1_only_flag = np.copy(coef_max)
    coef_l1_only_flag[:, l1_only_cut] = np.nan 
    coord_clean = np.copy(coord_max)
    coord_clean[:, :, cut] = np.nan
    coord_l1_flag = np.copy(coord_max)
    coord_l1_flag[:, :, l1_cut] = np.nan
    coord_l1_only_flag = np.copy(coord_max)
    coord_l1_only_flag[:, :, l1_only_cut] = np.nan
    del cut, l1_cut, l1_only_cut

    for p in range(2):
        map_r_z[r, :, p] = np.histogram(coord_max[p, 0], bins = z_bins)[0].astype(int)
        map_r_a[r, :, p] = np.histogram(coord_max[p, 1], bins = a_bins)[0].astype(int)
        map_r_c[r, :, p] = np.histogram(coef_max[p], bins = c_bins)[0].astype(int)
        map_r_z_l1[r, :, p] = np.histogram(coord_l1_flag[p, 0], bins = z_bins)[0].astype(int)
        map_r_a_l1[r, :, p] = np.histogram(coord_l1_flag[p, 1], bins = a_bins)[0].astype(int)
        map_r_c_l1[r, :, p] = np.histogram(coef_l1_flag[p], bins = c_bins)[0].astype(int)
        if b_runs[r]: continue
        map_r_z_cut[r, :, p] = np.histogram(coord_clean[p, 0], bins = z_bins)[0].astype(int)
        map_r_a_cut[r, :, p] = np.histogram(coord_clean[p, 1], bins = a_bins)[0].astype(int)
        map_r_c_cut[r, :, p] = np.histogram(coef_clean[p], bins = c_bins)[0].astype(int)
        map_r_z_cut_l1[r, :, p] = np.histogram(coord_l1_only_flag[p, 0], bins = z_bins)[0].astype(int)
        map_r_a_cut_l1[r, :, p] = np.histogram(coord_l1_only_flag[p, 1], bins = a_bins)[0].astype(int)
        map_r_c_cut_l1[r, :, p] = np.histogram(coef_l1_only_flag[p], bins = c_bins)[0].astype(int)        
    del coef_max, coord_max, coef_clean, coef_l1_flag, coef_l1_only_flag, coord_clean, coord_l1_flag, coord_l1_only_flag

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_l1_1d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('nan_counts', data=nan_counts, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z', data=map_r_z, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_l1', data=map_r_z_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut', data=map_r_z_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut_l1', data=map_r_z_cut_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a', data=map_r_a, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_l1', data=map_r_a_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut', data=map_r_a_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut_l1', data=map_r_a_cut_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c', data=map_r_c, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_l1', data=map_r_c_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut', data=map_r_c_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut_l1', data=map_r_c_cut_l1, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






