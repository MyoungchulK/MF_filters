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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Random_WF_New/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

d_old_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Random_WF_Old/'

# detector config
ant_num = antenna_info()[2]

diff_edge = 50
diff_range = np.arange(-1*diff_edge,diff_edge).astype(int)

diff_bins, diff_bin_center = bin_range_maker(diff_range, len(diff_range)*10)
diff_hist = np.full((len(diff_bin_center), 2, ant_num), 0, dtype = int)
print(diff_hist.shape)

for r in tqdm(range(len(d_run_tot))):
  #if r < 50: 
    if d_run_tot[r] in bad_runs:
        print('bad run:',d_list[r],d_run_tot[r])
        continue

    file_name = f'Random_WF_A{Station}_R{d_run_tot[r]}.h5'

    hf_new = h5py.File(d_list[r], 'r')
    new_evt_entry = hf_new['evt_entry'][:]       
        
    hf_old = h5py.File(d_old_path+file_name, 'r')
    old_evt_entry = hf_old['evt_entry'][:]

    evt_diff = new_evt_entry - old_evt_entry
    if len(np.where(np.abs(evt_diff) > 0)[0]) > 0:
        print('Evt Entry is different!')
        print(d_path+f'Random_WF_A{Station}_R{d_run_tot[r]}.h5')
        del hf_new, hf_old, evt_diff, new_evt_entry, old_evt_entry 
        continue
    del evt_diff, new_evt_entry, old_evt_entry      
 
    new_wf_all = hf_new['wf_all'][:]
    old_wf_all = hf_old['wf_all'][:]

    time_diff = new_wf_all[:,0] - old_wf_all[:,0]
    volt_diff = new_wf_all[:,1] - old_wf_all[:,1]
    del hf_new, hf_old, old_wf_all, new_wf_all

    for a in range(ant_num):
           
        time_ant_diff = time_diff[:,a]
        volt_ant_diff = volt_diff[:,a]
 
        time_ant_diff = time_ant_diff.flatten()
        volt_ant_diff = volt_ant_diff.flatten()

        diff_hist[:,0,a] += np.histogram(time_ant_diff, bins=diff_bins)[0]
        diff_hist[:,1,a] += np.histogram(volt_ant_diff, bins=diff_bins)[0]
        del time_ant_diff, volt_ant_diff

    del time_diff, volt_diff

print(diff_range.shape)
print(diff_bins.shape)
print(diff_bin_center.shape)
print(diff_hist.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Random_WF_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('diff_range', data=diff_range, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bins', data=diff_bins, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bin_center', data=diff_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('diff_hist', data=diff_hist, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















