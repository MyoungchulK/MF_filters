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

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

z_bins = np.linspace(0, 180, 180 + 1, dtype = int)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1, dtype = int)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)

map_cut = np.full((a_bin_len, z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config

debug = []

sh_path = '/home/mkim/analysis/MF_filters/scripts/reco_cal_debug.sh'
if not os.path.exists(sh_path):
    print(f'There is no {sh_path}. Making it!')
    cvm_path = 'source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh\n'
    with open(sh_path, 'w') as f:
        f.write(cvm_path)
else:
    print(f'There is {sh_path}. Moving on!')

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    evt = hf['evt_num'][:]
    del hf, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut'][:]
    qual[:, 14] = 0
    cut = np.in1d(evt, evt_full[np.nansum(qual, axis = 1) != 0])
    del q_name, hf_q, qual, evt_full

    coord[:,:,:,:,cut] = np.nan
    del cut

    if Station == 2:
        z_idx1 = np.logical_and(coord[0,0,0,0] > 52, coord[0,0,0,0] < 61)
        z_idx2 = np.logical_and(coord[0,0,0,0] > 107, coord[0,0,0,0] < 120)
        z_idx3 = np.logical_and(coord[0,0,0,0] > 80, coord[0,0,0,0] < 90)
        a_idx1 = np.logical_and(coord[0,1,0,0] > 150, coord[0,1,0,0] < 158)
        a_idx2 = np.logical_and(coord[0,1,0,0] > 238, coord[0,1,0,0] < 245)
        v_idx1 = np.logical_and(z_idx1, a_idx1)
        v_idx2 = np.logical_and(z_idx2, a_idx1)
        v_idx3 = np.logical_and(z_idx3, a_idx2)
        v_idx = np.any((v_idx1, v_idx2, v_idx3), axis = 0)
        del z_idx1, z_idx2, z_idx3, a_idx1, a_idx2, v_idx1, v_idx2, v_idx3

        z_idx1 = np.logical_and(coord[1,0,0,0] > 107, coord[1,0,0,0] < 120)
        a_idx1 = np.logical_and(coord[1,1,0,0] > 150, coord[1,1,0,0] < 158)
        h_idx1 = np.logical_and(z_idx1, a_idx1)
        z_idx2 = np.logical_and(coord[1,0,0,0] > 80, coord[1,0,0,0] < 90)
        a_idx2 = np.logical_and(coord[1,1,0,0] > 238, coord[1,1,0,0] < 245)
        h_idx2 = np.logical_and(z_idx2, a_idx2)
        h_idx = np.logical_or(h_idx1, h_idx2)
        del z_idx1, a_idx1, h_idx1, z_idx2, a_idx2, h_idx2
    if Station == 3:
        if g_idx < 5:
            z_idx1 = np.logical_and(coord[0,0,0,0] > 100, coord[0,0,0,0] < 110)
            a_idx1 = np.logical_and(coord[0,1,0,0] > 240, coord[0,1,0,0] < 246)
            v_idx = np.logical_and(z_idx1, a_idx1)
            z_idx2 = np.logical_and(coord[1,0,0,0] > 95, coord[1,0,0,0] < 105)
            a_idx2 = np.logical_and(coord[1,1,0,0] > 240, coord[1,1,0,0] < 246)
            h_idx = np.logical_and(z_idx2, a_idx2)
        else:
            z_idx1 = np.logical_and(coord[0,0,0,0] > 100, coord[0,0,0,0] < 112)
            a_idx1 = np.logical_and(coord[0,1,0,0] > -1, coord[0,1,0,0] < 361)
            v_idx = np.logical_and(z_idx1, a_idx1)
            z_idx2 = np.logical_and(coord[1,0,0,0] > 100, coord[1,0,0,0] < 112)
            a_idx2 = np.logical_and(coord[1,1,0,0] > -1, coord[1,1,0,0] < 361)
            h_idx = np.logical_and(z_idx2, a_idx2)
        del z_idx1, a_idx1, z_idx2, a_idx2
    coord[:,:,:,:,v_idx] = np.nan
    coord[:,:,:,:,h_idx] = np.nan

    no_nan_v = np.logical_and(~np.isnan(coord[0,0,0,0]), cal_t)
    no_nan_h = np.logical_and(~np.isnan(coord[1,0,0,0]), cal_t)
    if np.any(no_nan_v) or np.any(no_nan_h):
        v_tag = (no_nan_v).astype(int)
        h_tag = (no_nan_h).astype(int) * 2
        tot_tag = (v_tag + h_tag).astype(int)
        none_zero = tot_tag != 0
        de_evt = evt[none_zero]
        de_tag = tot_tag[none_zero]
        for d in range(len(de_evt)): 
            de_arr = np.array([Station, d_run_tot[r], de_evt[d], de_tag[d], g_idx+1], dtype = int)
            print(f'SHIT!!! {de_arr}')
            debug.append(de_arr)
            de_path = f'python3 -W ignore script_executor.py -k wf -s {Station} -r {d_run_tot[r]} -a {de_evt[d]}\n'
            with open(sh_path, 'a') as f:
                f.write(de_path)
        del v_tag, h_tag, tot_tag, none_zero
    del evt, no_nan_v, no_nan_h

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    map_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord[pol, 1, rad, sol][t_list[t]], coord[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
    del coord, t_list, rf_t, cal_t, soft_t

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_Cal_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_cut', data=map_cut, compression="gzip", compression_opts=9)
hf.create_dataset('debug', data=np.asarray(debug), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






