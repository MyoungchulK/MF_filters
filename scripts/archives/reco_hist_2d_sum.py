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

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/*Sur_Max_2d*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)

z_bins1 = np.linspace(90, -90, 180 + 1)
z_bin_center1 = (z_bins1[1:] + z_bins1[:-1]) / 2
a_bins1 = np.linspace(-180, 180, 360 + 1)
a_bin_center1 = (a_bins1[1:] + a_bins1[:-1]) / 2
a_bin_len = len(a_bin_center1)
z_bin_len = len(z_bin_center1)

map_az = np.full((a_bin_len, z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_az_cut = np.copy(map_az)
map_az_cut_cal = np.copy(map_az)
map_az_cut_cal_sur = np.copy(map_az)
map_ac = np.full((a_bin_len, c_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_ac_cut = np.copy(map_ac)
map_ac_cut_cal = np.copy(map_ac)
map_ac_cut_cal_sur = np.copy(map_ac)
map_zc = np.full((z_bin_len, c_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_zc_cut = np.copy(map_zc)
map_zc_cut_cal = np.copy(map_zc)
map_zc_cut_cal_sur = np.copy(map_zc)
map_zc_max = np.full((z_bin_len, c_bin_len, 3, num_configs), 0, dtype = int)
map_ac_max = np.full((a_bin_len, c_bin_len, 3, num_configs), 0, dtype = int)
map_az_max = np.full((a_bin_len, z_bin_len, 3, num_configs), 0, dtype = int)
del z_bin_len, a_bin_len, c_bin_len

for r in tqdm(range(len(d_run_tot))):
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    map_az += hf['map_az'][:]
    map_az_cut += hf['map_az_cut'][:]
    map_az_cut_cal += hf['map_az_cut_cal'][:]
    map_az_cut_cal_sur += hf['map_az_cut_cal_sur'][:]
    map_ac += hf['map_ac'][:]
    map_ac_cut += hf['map_ac_cut'][:]
    map_ac_cut_cal += hf['map_ac_cut_cal'][:]
    map_ac_cut_cal_sur += hf['map_ac_cut_cal_sur'][:]
    map_zc += hf['map_zc'][:]
    map_zc_cut += hf['map_zc_cut'][:]
    map_zc_cut_cal += hf['map_zc_cut_cal'][:]
    map_zc_cut_cal_sur += hf['map_zc_cut_cal_sur'][:]
    map_zc_max += hf['map_zc_max'][:]
    map_ac_max += hf['map_ac_max'][:]
    map_az_max += hf['map_az_max'][:]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_New_Cal_Sur_Max_2d_A{Station}_Rall.h5'
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
hf.create_dataset('map_az_cut_cal_sur', data=map_az_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac', data=map_ac, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut', data=map_ac_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut_cal', data=map_ac_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut_cal_sur', data=map_ac_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc', data=map_zc, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut', data=map_zc_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut_cal', data=map_zc_cut_cal, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut_cal_sur', data=map_zc_cut_cal_sur, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_max', data=map_zc_max, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_max', data=map_ac_max, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_max', data=map_az_max, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






