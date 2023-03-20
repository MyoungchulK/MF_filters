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

map_az = np.full((a_bin_len, z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_az_cut = np.copy(map_az)
map_az_cut_cal = np.copy(map_az)
map_ac = np.full((a_bin_len, c_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_ac_cut = np.copy(map_ac)
map_ac_cut_cal = np.copy(map_ac)
map_zc = np.full((z_bin_len, c_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_zc_cut = np.copy(map_zc)
map_zc_cut_cal = np.copy(map_zc)
map_zc_max = np.full((z_bin_len, c_bin_len, 3, num_configs), 0, dtype = int)
map_ac_max = np.full((a_bin_len, c_bin_len, 3, num_configs), 0, dtype = int)
map_az_max = np.full((a_bin_len, z_bin_len, 3, num_configs), 0, dtype = int)
del z_bin_len, a_bin_len, c_bin_len

def get_calpulser_cut(st, run):

    if st == 2:
        cp6 = np.full((2, 2), np.nan, dtype = float)
        cp6[0, 0] = -1.45
        cp6[0, 1] = 10.45
        cp6[1, 0] = 56.65
        cp6[1, 1] = 69.15
        cp5 = np.full((2, 2), np.nan, dtype = float)
        cp5[0, 0] = -28
        cp5[0, 1] = -19
        cp5[1, 0] = -29.35
        cp5[1, 1] = -21.75
        cp5_m = np.full((2, 2), np.nan, dtype = float)
        cp5_m[0, 0] = 27.15
        cp5_m[0, 1] = 37.75
        cp5_m[1, 0] = -29.35
        cp5_m[1, 1] = -21.75
        cp5_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_2020[0, 0] = -27.85
        cp5_2020[0, 1] = -20.25
        cp5_2020[1, 0] = -31.05
        cp5_2020[1, 1] = -21.65
        cp5_m_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_m_2020[0, 0] = 28.65
        cp5_m_2020[0, 1] = 37.15
        cp5_m_2020[1, 0] = -45.85
        cp5_m_2020[1, 1] = -38.35

        if run < 1901:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 1900 and run < 1935:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
        elif run == 1935:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
            cp_cut[1] = cp6
        elif run > 1935 and run < 7006:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 7005 and run < 8098:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
            cp_cut[1] = cp5_m
        elif run > 8097 and run < 9505:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 9504 and run < 15527:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
        elif run > 15526:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2020
            cp_cut[1] = cp5_m_2020
        num_cuts = cp_cut.shape[0]

    elif st == 3:
        cp6 = np.full((2, 2), np.nan, dtype = float)
        cp6[0, 0] = -16.75
        cp6[0, 1] = -12.25
        cp6[1, 0] = 61.25
        cp6[1, 1] = 65.75
        cp5_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_2020[0, 0] = -18.25
        cp5_2020[0, 1] = -12.65
        cp5_2020[1, 0] = -27.65
        cp5_2020[1, 1] = -18.45
        cp5_2019 = np.full((2, 2), np.nan, dtype = float)
        cp5_2019[0, 0] = -18.25
        cp5_2019[0, 1] = -12.65
        cp5_2019[1, 0] = -35.05
        cp5_2019[1, 1] = 1.75
        cp6_m_2019 = np.full((2, 2), np.nan, dtype = float)
        cp6_m_2019[0, 0] = -21.25
        cp6_m_2019[0, 1] = -16.35
        cp6_m_2019[1, 0] = -117.75
        cp6_m_2019[1, 1] = -113.95

        if run < 12873:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 12872 and run < 13901:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
            cp_cut[1] = cp6_m_2019
        elif run > 13900 and run < 16487:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2019
        elif run > 16486:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2020
        num_cuts = cp_cut.shape[0]

    return cp_cut, num_cuts

def get_calpulser_pol(st, run):

    pol_idx = 0
    if st == 2 and (run > 1877 and run < 1887):
        pol_idx = 1
    elif st == 3 and (run > 923 and run < 934):
        pol_idx = 1

    return pol_idx

runs = np.copy(d_run_tot)
b_runs = np.in1d(runs, bad_runs).astype(int)
del bad_runs, runs

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    config = hf['config'][2]
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
    del q_name, hf_q, qual, evt_full

    coord_cut = np.copy(coord)
    coord_cut[:, :, :, :, cut] = np.nan
    coord_cut_cal = np.copy(coord_cut)
    coef_cut = np.copy(coef)
    coef_cut[:, :, :, cut] = np.nan
    coef_cut_cal = np.copy(coef_cut)

    pol_idx = get_calpulser_pol(Station, d_run_tot[r])
    cp_cut, num_cuts = get_calpulser_cut(Station, d_run_tot[r])
    cal_cut = np.full((len(evt)), False, dtype = bool)
    for c in range(num_cuts):
        ele_flag = np.digitize(89.5 - coord_cut_cal[pol_idx, 0, 0, 0], cp_cut[c, 0]) == 1
        azi_flag = np.digitize(coord_cut_cal[pol_idx, 1, 0, 0] - 179.5, cp_cut[c, 1]) == 1
        cal_cut += np.logical_and(ele_flag, azi_flag)
        del ele_flag, azi_flag
    coord_cut_cal[:, :, :, :, cal_cut] = np.nan
    coef_cut_cal[:, :, :, cal_cut] = np.nan
    del pol_idx, cp_cut, num_cuts, evt

    g_idx = int(config - 1)
    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):
                    map_az[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord[pol, 1, rad, sol][t_list[t]], coord[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_ac[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord[pol, 1, rad, sol][t_list[t]], coef[pol, rad, sol][t_list[t]], bins = (a_bins, c_bins))[0].astype(int)
                    map_zc[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord[pol, 0, rad, sol][t_list[t]], coef[pol, rad, sol][t_list[t]], bins = (z_bins, c_bins))[0].astype(int)
                    if b_runs[r]: continue
                    map_az_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut[pol, 1, rad, sol][t_list[t]], coord_cut[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_ac_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut[pol, 1, rad, sol][t_list[t]], coef_cut[pol, rad, sol][t_list[t]], bins = (a_bins, c_bins))[0].astype(int)
                    map_zc_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut[pol, 0, rad, sol][t_list[t]], coef_cut[pol, rad, sol][t_list[t]], bins = (z_bins, c_bins))[0].astype(int)
                    map_az_cut_cal[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut_cal[pol, 1, rad, sol][t_list[t]], coord_cut_cal[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_ac_cut_cal[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut_cal[pol, 1, rad, sol][t_list[t]], coef_cut_cal[pol, rad, sol][t_list[t]], bins = (a_bins, c_bins))[0].astype(int)
                    map_zc_cut_cal[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord_cut_cal[pol, 0, rad, sol][t_list[t]], coef_cut_cal[pol, rad, sol][t_list[t]], bins = (z_bins, c_bins))[0].astype(int)

    if b_runs[r]: continue
    nan_idx = ~np.logical_or(cut, cal_cut)
    rf_idx = np.logical_and(rf_t, nan_idx)
    coef_rf = coef_cut_cal[:, :, :, rf_idx]
    
    coef_rf_v = coef_rf[0]
    coef_rf_h = coef_rf[1]
    coef_rf_rh = np.reshape(coef_rf, (8, -1))
    coef_rf_v_rh = np.reshape(coef_rf_v, (4, -1))
    coef_rf_h_rh = np.reshape(coef_rf_h, (4, -1))
    coef_rf_re = np.nanargmax(coef_rf_rh, axis = 0)
    coef_rf_v_re = np.nanargmax(coef_rf_v_rh, axis = 0)
    coef_rf_h_re = np.nanargmax(coef_rf_h_rh, axis = 0)

    rf_len = np.count_nonzero(rf_idx)
    coef_rf_max = np.full((rf_len), np.nan, dtype = float)
    coef_rf_v_max = np.full((rf_len), np.nan, dtype = float)
    coef_rf_h_max = np.full((rf_len), np.nan, dtype = float)

    coord_rf = coord_cut_cal[:, :, :, :, rf_idx]
    coord_rf_v = np.reshape(coord_rf[0], (2, 4, -1))
    coord_rf_h = np.reshape(coord_rf[1], (2, 4, -1))
    coord_rf_trans = np.transpose(coord_rf, (1,0,2,3,4))
    coord_rf_trans = np.reshape(coord_rf_trans, (2, 8, -1))
    coord_rf_tp = np.full((2, rf_len), np.nan, dtype = float)
    coord_rf_v_tp = np.full((2, rf_len), np.nan, dtype = float)
    coord_rf_h_tp = np.full((2, rf_len), np.nan, dtype = float)
    for rr in range(rf_len):
        coef_rf_max[rr] = coef_rf_rh[coef_rf_re[rr], rr]
        coef_rf_v_max[rr] = coef_rf_v_rh[coef_rf_v_re[rr], rr]
        coef_rf_h_max[rr] = coef_rf_h_rh[coef_rf_h_re[rr], rr]
        coord_rf_tp[:, rr] = coord_rf_trans[:, coef_rf_re[rr], rr]
        coord_rf_v_tp[:, rr] = coord_rf_v[:, coef_rf_v_re[rr], rr]
        coord_rf_h_tp[:, rr] = coord_rf_h[:, coef_rf_h_re[rr], rr]
 
    map_zc_max[:, :, 0, g_idx] += np.histogram2d(coord_rf_tp[0], coef_rf_max, bins = (z_bins, c_bins))[0].astype(int)
    map_zc_max[:, :, 1, g_idx] += np.histogram2d(coord_rf_v_tp[0], coef_rf_v_max, bins = (z_bins, c_bins))[0].astype(int)
    map_zc_max[:, :, 2, g_idx] += np.histogram2d(coord_rf_h_tp[0], coef_rf_h_max, bins = (z_bins, c_bins))[0].astype(int)
    map_ac_max[:, :, 0, g_idx] += np.histogram2d(coord_rf_tp[1], coef_rf_max, bins = (a_bins, c_bins))[0].astype(int)
    map_ac_max[:, :, 1, g_idx] += np.histogram2d(coord_rf_v_tp[1], coef_rf_v_max, bins = (a_bins, c_bins))[0].astype(int)
    map_ac_max[:, :, 2, g_idx] += np.histogram2d(coord_rf_h_tp[1], coef_rf_h_max, bins = (a_bins, c_bins))[0].astype(int)
    map_az_max[:, :, 0, g_idx] += np.histogram2d(coord_rf_tp[1], coord_rf_tp[0], bins = (a_bins, z_bins))[0].astype(int)
    map_az_max[:, :, 1, g_idx] += np.histogram2d(coord_rf_v_tp[1], coord_rf_v_tp[0], bins = (a_bins, z_bins))[0].astype(int)
    map_az_max[:, :, 2, g_idx] += np.histogram2d(coord_rf_h_tp[1], coord_rf_h_tp[0], bins = (a_bins, z_bins))[0].astype(int)
    del g_idx, coef, coef_cut, coord, coord_cut, t_list, rf_t, cal_t, soft_t, coord_cut_cal, coef_cut_cal, coef_rf, coef_rf_v, coef_rf_h, coef_rf_re, coef_rf_v_re, coef_rf_h_re
    del coef_rf_max, coef_rf_v_max, coef_rf_h_max, coord_rf, coord_rf_v, coord_rf_h

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_New_Cal_Max_2d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_az', data=map_az, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_cut', data=map_az_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_cut_cal', data=map_az_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac', data=map_ac, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut', data=map_ac_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut_cal', data=map_ac_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc', data=map_zc, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut', data=map_zc_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut_cal', data=map_zc_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_max', data=map_zc_max, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_max', data=map_ac_max, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_max', data=map_az_max, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






