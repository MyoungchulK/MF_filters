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
#from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import get_path_info_v2

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f
sim_type = str(sys.argv[4])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_sim/*{sim_type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
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

num_fs = 3
map_az = np.full((num_fs, num_configs, a_bin_len, z_bin_len, 2, 2, 2), 0, dtype = int) # flavor, configs, a, z, trig, pol, rad, sol, config
map_ac = np.full((num_fs, num_configs, a_bin_len, c_bin_len, 2, 2, 2), 0, dtype = int) 
map_zc = np.full((num_fs, num_configs, z_bin_len, c_bin_len, 2, 2, 2), 0, dtype = int)
del z_bin_len, a_bin_len, c_bin_len

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:] # pol, rad, sol, evt
    cons = hf['config'][:]
    del hf

    config = cons[2] - 1
    if sim_type == 'signal':
        flavor = cons[4] - 1
    else:
        flavor = 0

    for pol in range(2):
            for rad in range(2):
                for sol in range(2):
                    map_az[flavor, config, :, :, pol, rad, sol] += np.histogram2d(coord[pol, 1, rad, sol], coord[pol, 0, rad, sol], bins = (a_bins, z_bins))[0].astype(int)
                    map_ac[flavor, config, :, :, pol, rad, sol] += np.histogram2d(coord[pol, 1, rad, sol], coef[pol, rad, sol], bins = (a_bins, c_bins))[0].astype(int)
                    map_zc[flavor, config, :, :, pol, rad, sol] += np.histogram2d(coord[pol, 0, rad, sol], coef[pol, rad, sol], bins = (z_bins, c_bins))[0].astype(int)
    del cons, coef, coord, config, flavor

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Sim_Map_New_Pad_2d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_az', data=map_az, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac', data=map_ac, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc', data=map_zc, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






