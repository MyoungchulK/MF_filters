import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
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
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
livetime = np.full((d_len, 3), 0, dtype = float)
nan_counts = np.full((d_len), 0, dtype = int)

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
c_bins = np.linspace(0, 1.2, 1200 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
m_bins = np.linspace(0, 100, 200 + 1)
m_bin_center = (m_bins[1:] + m_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bin_len = len(a_bin_center)
c_bin_len = len(c_bin_center)
m_bin_len = len(m_bin_center)

map_r_z = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a, trig, pol, rad, sol
map_r_z_cut = np.copy(map_r_z)
map_r_z_rf = np.full((d_len, z_bin_len, 3, 2), 0, dtype = int)
map_r_a = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a_cut = np.copy(map_r_a)
map_r_a_rf = np.full((d_len, a_bin_len, 3, 2), 0, dtype = int)
map_r_c = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a or c, trig, pol, rad, sol
map_r_c_cut = np.copy(map_r_c)
map_r_c_rf = np.full((d_len, c_bin_len, 3, 2), 0, dtype = int)
mf_r_c = np.full((d_len, m_bin_len, 3, 2), 0, dtype = int) # run, z or a or c, trig, pol, rad, sol
mf_r_c_cut = np.copy(mf_r_c)
#mf_r_c_rf = np.full((d_len, m_bin_len, 3, 2), 0, dtype = int)
map_az = np.full((num_configs, a_bin_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_az_cut = np.copy(map_az)
map_az_rf = np.full((num_configs, a_bin_len, z_bin_len, 3, 2), 0, dtype = int)
del bad_runs, z_bin_len, a_bin_len, c_bin_len

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs[r] = hf['config'][2]
    con_idx = int(configs[r] - 1)
    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:] # pol, rad, sol, evt
    evt = hf['evt_num'][:]
    num_evts = len(evt)
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    del hf, trig

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
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
    del q_name, hf_q, qual, evt_full, tot_live, bad_live, good_live, evt

    m_name = f'{m_path}mf_A{Station}_R{d_run_tot[r]}.h5'
    try:
        hf = h5py.File(m_name, 'r')
    except OSError:
        print(d_list[r])
        continue
    mf_m = hf['mf_max'][:]
    mf_t = hf['mf_temp'][:, 1:3]

    mf_m_cut = np.copy(mf_m)
    mf_m_cut[:, cut] = np.nan
    for t in range(3):
        for pol in range(2):
            mf_r_c[r, :, t, pol] = np.histogram(mf_m[pol][t_list[t]], bins = m_bins)[0].astype(int)
            if b_runs[r]: continue
            mf_r_c_cut[r, :, t, pol] = np.histogram(mf_m_cut[pol][t_list[t]], bins = m_bins)[0].astype(int)
            #if t == 0:

    del hf, mf_m, mf_t, mf_m_cut

    coord_cut = np.copy(coord)
    coord_cut[:, :, :, :, cut] = np.nan
    coef_cut = np.copy(coef)
    coef_cut[:, :, :, cut] = np.nan
    del cut

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    map_r_z[r, :, t, pol, rad, sol] = np.histogram(coord[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a[r, :, t, pol, rad, sol] = np.histogram(coord[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c[r, :, t, pol, rad, sol] = np.histogram(coef[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)
                    map_az[con_idx, :, :, t, pol, rad, sol] += np.histogram2d(coord[pol, 1, rad, sol][t_list[t]], coord[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    if b_runs[r]: continue
                    map_r_z_cut[r, :, t, pol, rad, sol] = np.histogram(coord_cut[pol, 0, rad, sol][t_list[t]], bins = z_bins)[0].astype(int)
                    map_r_a_cut[r, :, t, pol, rad, sol] = np.histogram(coord_cut[pol, 1, rad, sol][t_list[t]], bins = a_bins)[0].astype(int)
                    map_r_c_cut[r, :, t, pol, rad, sol] = np.histogram(coef_cut[pol, rad, sol][t_list[t]], bins = c_bins)[0].astype(int)
                    map_az_cut[con_idx, :, :, t, pol, rad, sol] += np.histogram2d(coord_cut[pol, 1, rad, sol][t_list[t]], coord_cut[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)

    if b_runs[r]: continue
    coef_re = np.reshape(coef_cut, (2, 4, -1))
    coef_max = np.nanmax(coef_re, axis = 1)
    coord_re = np.reshape(coord_cut, (2, 2, 4, -1))
    coord_max = np.full((2, 2, num_evts), np.nan, dtype = float)
    counts = 0
    for e in range(num_evts):
        try:
            coef_max_idx = np.nanargmax(coef_re[:, :, e], axis = 1)
        except ValueError:
            counts += 1
            continue
        coord_max[0, :, e] = coord_re[0, :, coef_max_idx[0], e]
        coord_max[1, :, e] = coord_re[1, :, coef_max_idx[1], e]
        del coef_max_idx
    nan_counts[r] = counts

    for t in range(3):
        for pol in range(2):
            map_r_z_rf[r, :, t, pol] = np.histogram(coord_max[pol, 0][t_list[t]], bins = z_bins)[0].astype(int)
            map_r_a_rf[r, :, t, pol] = np.histogram(coord_max[pol, 1][t_list[t]], bins = a_bins)[0].astype(int)
            map_r_c_rf[r, :, t, pol] = np.histogram(coef_max[pol][t_list[t]], bins = c_bins)[0].astype(int)
            map_az_rf[con_idx, :, :, t, pol] += np.histogram2d(coord_max[pol, 1][t_list[t]], coord_max[pol, 0][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)        
    del con_idx, coord, coef, rf_t, cal_t, soft_t, t_list, coord_cut, coef_cut, num_evts, coef_re, coef_max, coord_re, counts

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Signal_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('m_bins', data=m_bins, compression="gzip", compression_opts=9)
hf.create_dataset('m_bin_center', data=m_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('nan_counts', data=nan_counts, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z', data=map_r_z, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut', data=map_r_z_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_rf', data=map_r_z_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a', data=map_r_a, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut', data=map_r_a_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_rf', data=map_r_a_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c', data=map_r_c, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut', data=map_r_c_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_rf', data=map_r_c_rf, compression="gzip", compression_opts=9)
hf.create_dataset('mf_r_c', data=mf_r_c, compression="gzip", compression_opts=9)
hf.create_dataset('mf_r_c_cut', data=mf_r_c_cut, compression="gzip", compression_opts=9)
#hf.create_dataset('mf_r_c_rf', data=mf_r_c_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_az', data=map_az, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_cut', data=map_az_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_rf', data=map_az_rf, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






