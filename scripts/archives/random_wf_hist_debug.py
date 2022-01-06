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
  #if r == 443:
  #if d_run_tot[r] == 1125: 
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
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

    ant = np.array([0,4,8])
 
    new_wf_all = hf_new['wf_all'][:,1,ant]
    old_wf_all = hf_old['wf_all'][:,1,ant]
    """
    volt_diff = np.array([])
    for a in range(100):
     #if a == 95:
      new_indi = new_wf_all[:,a]
      old_indi = old_wf_all[:,a]

      print(new_indi)
      print(old_indi)

      new_indi1 = new_indi[~np.isnan(new_indi)]
      old_indi1 = old_indi[~np.isnan(old_indi)]

      print(new_indi1)
      print(old_indi1)

      volt_diff = np.append(volt_diff, new_indi1 - old_indi1)

      print(a,volt_diff)

    volt_diff = volt_diff.flatten()
    """
    new_wf_all1 = new_wf_all.flatten()
    old_wf_all1 = old_wf_all.flatten()

    new_wf_all2 = new_wf_all1[~np.isnan(new_wf_all1)]
    old_wf_all2 = old_wf_all1[~np.isnan(old_wf_all1)]

    volt_diff = new_wf_all2 - old_wf_all2
    if len(np.where(np.abs(volt_diff)!=0)[0])>0:
        print('Wrong!!')
        print(np.where(np.abs(volt_diff)!=0)[0])
        print(volt_diff[np.where(np.abs(volt_diff)!=0)[0]])
        print(new_wf_all2[np.where(np.abs(volt_diff)!=0)[0]])
        print(old_wf_all2[np.where(np.abs(volt_diff)!=0)[0]])
        print(d_list[r])
        print(d_old_path+file_name) 

    del hf_new, hf_old, old_wf_all, new_wf_all

print(diff_range.shape)
print(diff_bins.shape)
print(diff_bin_center.shape)
print(diff_hist.shape)





















