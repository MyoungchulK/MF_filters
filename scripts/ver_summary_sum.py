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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Ver_A{Station}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

hf = h5py.File(d_list[0], 'r')
runs = hf['runs'][:]
b_runs = hf['b_runs'][:]
configs = hf['configs'][:]
run_ep = hf['run_ep'][:]
evt_ep = hf['evt_ep'][:]
trig_ep = hf['trig_ep'][:]
con_ep = hf['con_ep'][:]
unix_ep = hf['unix_ep'][:]
del hf

theta = np.full((3, 0), np.nan, dtype = float) # ch, evts
phi = np.copy(theta)
r = np.copy(theta)
x = np.copy(theta)
y = np.copy(theta)
z = np.copy(theta)
nhits = np.copy(theta)

for rr in tqdm(range(len(d_run_tot))):
    
  #if rr <10:

    try:
        hf = h5py.File(d_list[rr], 'r')
    except OSError: 
        print(d_list[rr])
        continue

    theta1 = hf['theta'][:]
    phi1 = hf['phi'][:]
    r1 = hf['r'][:]
    x1 = hf['x'][:]
    y1 = hf['y'][:]
    z1 = hf['z'][:]
    nhits1 = hf['nhits'][:]

    theta = np.concatenate((theta, theta1), axis = 1)
    phi = np.concatenate((phi, phi1), axis = 1)
    r = np.concatenate((r, r1), axis = 1)
    x = np.concatenate((x, x1), axis = 1)
    y = np.concatenate((y, y1), axis = 1)
    z = np.concatenate((z, z1), axis = 1)
    nhits = np.concatenate((nhits, nhits1), axis = 1)
    del hf, theta1, phi1, r1, x1, y1, z1, nhits1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Ver_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('theta', data=theta, compression="gzip", compression_opts=9)
hf.create_dataset('phi', data=phi, compression="gzip", compression_opts=9)
hf.create_dataset('r', data=r, compression="gzip", compression_opts=9)
hf.create_dataset('x', data=x, compression="gzip", compression_opts=9)
hf.create_dataset('y', data=y, compression="gzip", compression_opts=9)
hf.create_dataset('z', data=z, compression="gzip", compression_opts=9)
hf.create_dataset('nhits', data=nhits, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')






