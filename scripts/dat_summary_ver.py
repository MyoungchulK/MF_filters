import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

known_issue = known_issue_loader(Station, verbose = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
v_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/vertex/'
del d_run_range

snr_ver = np.full((16, 0), 0, dtype = float)
coord_ver = np.full((2, 3, 0), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:   

    bad_ant = known_issue.get_bad_antenna(d_run_tot[r])
 
    v_name = f'{v_path}vertex_A{Station}_R{d_run_tot[r]}.h5'
    hf_v = h5py.File(v_name, 'r')
    snr = hf_v['snr'][:]
    snr[bad_ant] = np.nan
    snr_ver = np.concatenate((snr_ver, snr), axis = 1)
    theta = hf_v['theta'][:]
    phi = hf_v['phi'][:]
    coord = np.asarray([theta, phi])
    coord_ver = np.concatenate((coord_ver, coord), axis = 2)
    del v_name, hf_v, theta, phi, coord, snr, bad_ant

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Ver_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('snr_ver', data=snr_ver, compression="gzip", compression_opts=9)
hf.create_dataset('coord_ver', data=coord_ver, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






