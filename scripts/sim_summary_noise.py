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

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_sim/*noise*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

num_evts = 1000
sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
flavor = np.copy(sim_run)
coef = np.full((d_len, 2, 2, 2, num_evts), np.nan, dtype = float)
coord = np.full((d_len, 2, 2, 2, 2, num_evts), np.nan, dtype = float) 
z_bins = np.linspace(90, -90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    flavor[r] = cons[4]
    coef[r] = hf['coef'][:]
    coord[r] = hf['coord'][:]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_Noise_Pad_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




