import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run
from tools.run import file_sorter
from tools.antenna import antenna_info
from tools.run import config_checker

Station = int(sys.argv[1])
Config = int(sys.argv[2])

# bad runs
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

# sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Vol_Calib/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# detector config
cal_type = 2
num_Ants = antenna_info()[2]
samp_per_block = 64
block_per_dda = 512
block_range = np.arange(block_per_dda)
nsamp = block_per_dda * samp_per_block
nsamp_range = np.arange(nsamp)
amp_range = np.arange(-120,120)
del samp_per_block, block_per_dda, nsamp

#config_type = 7

# config array
roll_mean_all = np.full((len(nsamp_range), len(amp_range), num_Ants, cal_type), 0, dtype = int)
block_all = np.full((len(block_range), len(amp_range), num_Ants, cal_type), 0, dtype = int)
print(roll_mean_all.shape)
print(block_all.shape)

for r in tqdm(range(len(d_run_tot))):
  #if r < 2: 
    if d_run_tot[r] in bad_runs:
        print('bad run:',d_list[r],d_run_tot[r])
        continue

    config = config_checker(Station, d_run_tot[r])
    if config != Config:
        continue
        
    hf = h5py.File(d_list[r], 'r')
    roll_mean_all[:,:,:,:] += hf['roll_mean_all'][:]
    block_all[:,:,:,:] += hf['block_all'][:]
    del hf, config
        
path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Vol_Calib_Hist_A{Station}_C{Config}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('roll_mean_all', data=roll_mean_all, compression="gzip", compression_opts=9)
hf.create_dataset('block_all', data=block_all, compression="gzip", compression_opts=9)
hf.create_dataset('nsamp_range', data=nsamp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('block_range', data=block_range, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















