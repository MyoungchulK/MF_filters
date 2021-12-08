import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run
from tools.run import file_sorter
from tools.antenna import antenna_info
from tools.run import bin_range_maker

Station = int(sys.argv[1])

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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/RMS_Peak_New/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

d_old_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/RMS_Peak_Old/'

# detector config
ant_num = antenna_info()[2]

diff_edge = 50
diff_range = np.arange(-1*diff_edge,diff_edge).astype(int)

diff_bins, diff_bin_center = bin_range_maker(diff_range, len(diff_range)*10)
peak_hist = np.full((len(diff_bin_center), 2, ant_num), 0, dtype = int)
rms_hist = np.full((len(diff_bin_center), ant_num), 0, dtype = int)
print(peak_hist.shape)
print(rms_hist.shape)

for r in tqdm(range(len(d_run_tot))):
  #if r < 50: 
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    file_name = f'RMS_Peak_A{Station}_R{d_run_tot[r]}.h5'
    try:
        hf_new = h5py.File(d_list[r], 'r')
        hf_old = h5py.File(d_old_path+file_name, 'r')
    except FileNotFoundError:
        print(f'No file for A{Station} R{d_run_tot[r]} !!!!!!!!!!!!!!')
        continue
    
    new_peak_all = hf_new['peak_all'][:]
    old_peak_all = hf_old['peak_all'][:]
    
    new_rms_all = hf_new['rms_all'][:]
    old_rms_all = hf_old['rms_all'][:]

    vtime_diff = new_peak_all[0] - old_peak_all[0]
    vpeak_diff = new_peak_all[1] - old_peak_all[1]

    rms_diff = new_rms_all - old_rms_all
    del hf_new, hf_old, old_peak_all, new_peak_all, old_rms_all, new_rms_all

    for a in range(ant_num):

        if len(np.where(vtime_diff[a] != 0)[0]) > 0:
            print(d_list[r])
            print(d_old_path+file_name)
            print('evt entry:',np.where(vtime_diff[a] != 0))
            print('time diff:',vtime_diff[a,np.where(vtime_diff[a] != 0)[0][0]])
        
        peak_hist[:,0,a] += np.histogram(vtime_diff[a], bins=diff_bins)[0]
        peak_hist[:,1,a] += np.histogram(vpeak_diff[a], bins=diff_bins)[0]

        rms_hist[:,a] += np.histogram(rms_diff[a], bins=diff_bins)[0]
        
    del vtime_diff, vpeak_diff, rms_diff

print(diff_range.shape)
print(diff_bins.shape)
print(diff_bin_center.shape)
print(peak_hist.shape)
print(rms_hist.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'RMS_Peak_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('diff_range', data=diff_range, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bins', data=diff_bins, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bin_center', data=diff_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('peak_hist', data=peak_hist, compression="gzip", compression_opts=9)
hf.create_dataset('rms_hist', data=rms_hist, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')




















