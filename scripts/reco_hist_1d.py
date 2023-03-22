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
from tools.ara_quality_cut import get_calpulser_cut


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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'
b_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/'
del d_run_range

z_bins = np.linspace(0, 180, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)

z_bins1 = np.linspace(90, -90, 180 + 1)
z_bin_center1 = (z_bins1[1:] + z_bins1[:-1]) / 2
a_bins1 = np.linspace(-180, 180, 360 + 1)
a_bin_center1 = (a_bins1[1:] + a_bins1[:-1]) / 2

map_r_z = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a, trig, pol, rad, sol
map_r_z_cut = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_z_cut_cal = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_z_cut_cal_sur = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a_cut = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a_cut_cal = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a_cut_cal_sur = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_c = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a or c, trig, pol, rad, sol
map_r_c_cut = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_c_cut_cal = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_c_cut_cal_sur = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_c_max = np.full((d_len, c_bin_len, 3), 0, dtype = int) # both, v, h
runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
years = np.copy(configs)
livetime = np.full((d_len, 3), 0, dtype = float)
del bad_runs

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
    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:] # pol, rad, sol, evt
    evt = hf['evt_num'][:]
    del hf

    b_name = f'{b_path}snr_A{Station}_R{d_run_tot[r]}.h5'
    hf_b = h5py.File(b_name, 'r')
    trig = hf_b['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    del b_name, hf_b, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    cut = np.in1d(evt, evt_full[qual])
    tot_live = np.nansum(hf_q['tot_qual_live_time'][:])
    bad_live = np.nansum(hf_q['tot_qual_sum_bad_live_time'][:])
    good_live = tot_live - bad_live
    livetime[r, 0] = tot_live
    livetime[r, 1] = good_live
    livetime[r, 2] = bad_live
    del q_name, hf_q, qual, evt_full, tot_live, bad_live, good_live

    coord_cut = np.copy(coord)
    coord_cut[:, :, :, :, cut] = np.nan
    coord_cut_cal = np.copy(coord_cut)
    coef_cut = np.copy(coef)
    coef_cut[:, :, :, cut] = np.nan
    coef_cut_cal = np.copy(coef_cut)
    del cut

    cp_cut, num_cuts, pol_idx = get_calpulser_cut(Station, d_run_tot[r])
    cal_cut = np.full((len(evt)), False, dtype = bool)
    for c in range(num_cuts):
        ele_flag = np.digitize(89.5 - coord_cut_cal[pol_idx, 0, 0, 0], cp_cut[c, 0]) == 1
        azi_flag = np.digitize(coord_cut_cal[pol_idx, 1, 0, 0] - 179.5, cp_cut[c, 1]) == 1
        cal_cut += np.logical_and(ele_flag, azi_flag)
        del ele_flag, azi_flag
    coord_cut_cal[:, :, :, :, cal_cut] = np.nan
    coef_cut_cal[:, :, :, cal_cut] = np.nan
    del con, config, pol_idx, cp_cut, num_cuts, cal_cut, evt

    coord_cut_cal_sur = np.copy(coord_cut_cal)
    coef_cut_cal_sur = np.copy(coef_cut_cal)
    scut_val = 35
    zenith_deg = 89.5 - coord_cut_cal_sur[:, 0, 1, :, :] # pol, thetaphi, rad, sol, evt
    zenith_deg = np.reshape(zenith_deg, (4, -1))
    scut = np.any(zenith_deg > scut_val, axis = 0)
    coord_cut_cal_sur[:, :, :, :, scut] = np.nan
    coef_cut_cal_sur[:, :, :, scut] = np.nan

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    map_r_z[r, :, t, pol, rad, sol] = np.histogram(coord[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a[r, :, t, pol, rad, sol] = np.histogram(coord[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c[r, :, t, pol, rad, sol] = np.histogram(coef[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)
                    if b_runs[r]: continue
                    map_r_z_cut[r, :, t, pol, rad, sol] = np.histogram(coord_cut[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a_cut[r, :, t, pol, rad, sol] = np.histogram(coord_cut[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c_cut[r, :, t, pol, rad, sol] = np.histogram(coef_cut[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)
                    map_r_z_cut_cal[r, :, t, pol, rad, sol] = np.histogram(coord_cut_cal[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a_cut_cal[r, :, t, pol, rad, sol] = np.histogram(coord_cut_cal[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c_cut_cal[r, :, t, pol, rad, sol] = np.histogram(coef_cut_cal[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)
                    map_r_z_cut_cal_sur[r, :, t, pol, rad, sol] = np.histogram(coord_cut_cal_sur[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a_cut_cal_sur[r, :, t, pol, rad, sol] = np.histogram(coord_cut_cal_sur[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c_cut_cal_sur[r, :, t, pol, rad, sol] = np.histogram(coef_cut_cal_sur[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)

    if b_runs[r]: continue
    coef_rf = coef_cut_cal_sur[:, :, :, rf_t]
    coef_rf_v = coef_rf[0]
    coef_rf_h = coef_rf[1]
    coef_rf_re = np.nanmax(np.reshape(coef_rf, (8, -1)), axis = 0)
    coef_rf_v_re = np.nanmax(np.reshape(coef_rf_v, (4, -1)), axis = 0)
    coef_rf_h_re = np.nanmax(np.reshape(coef_rf_h, (4, -1)), axis = 0)
    map_r_c_max[r, :, 0] = np.histogram(coef_rf_re, bins = c_bins)[0].astype(int)
    map_r_c_max[r, :, 1] = np.histogram(coef_rf_v_re, bins = c_bins)[0].astype(int)
    map_r_c_max[r, :, 2] = np.histogram(coef_rf_h_re, bins = c_bins)[0].astype(int)
    del coef, coef_cut, coord, coord_cut, t_list, rf_t, cal_t, soft_t, coord_cut_cal, coef_cut_cal, coef_rf, coef_rf_v, coef_rf_h, coef_rf_re, coef_rf_v_re, coef_rf_h_re

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_New_Cal_Sur_Max_1d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('years', data=years, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z', data=map_r_z, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut', data=map_r_z_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut_cal', data=map_r_z_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut_cal_sur', data=map_r_z_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a', data=map_r_a, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut', data=map_r_a_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut_cal', data=map_r_a_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut_cal_sur', data=map_r_a_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c', data=map_r_c, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut', data=map_r_c_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut_cal', data=map_r_c_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut_cal_sur', data=map_r_c_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_max', data=map_r_c_max, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






