import numpy as np
import os, sys
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.utility import size_checker
from tools.ara_quality_cut import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sensor_info/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
run_arr = []
atri_volt = []
atri_curr = []
dda_volt = []
dda_curr = []
dda_temp = []
tda_volt = []
tda_curr = []
tda_temp = []

bin_range = np.arange(0,20,0.01)
bins = np.linspace(0, 20, 2000+1)
temp_bins = np.linspace(-20, 20, 2000+1)
bin_center = (bins[1:] + bins[:-1]) * 0.5

for r in tqdm(range(len(d_run_tot))):

 #if r <10:
    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    dda_volt_evt = hf['dda_volt'][:]
    dda_curr_evt = hf['dda_curr'][:]
    dda_temp_evt = hf['dda_temp'][:]
    tda_volt_evt = hf['tda_volt'][:]
    tda_curr_evt = hf['tda_curr'][:]
    tda_temp_evt = hf['tda_temp'][:]

    run_arr.append(d_run_tot[r])

    atri_volt_hist = np.histogram(hf['atri_volt'][:], bins = bins)[0].astype(int)
    atri_curr_hist = np.histogram(hf['atri_curr'][:], bins = bins)[0].astype(int)
    atri_volt.append(atri_volt_hist)
    atri_curr.append(atri_curr_hist)

    dda_volt_run = np.full((len(bin_center), 4), 0, dtype = int)
    dda_curr_run = np.copy(dda_volt_run)
    dda_temp_run = np.copy(dda_volt_run)
    tda_volt_run = np.copy(dda_volt_run)
    tda_curr_run = np.copy(dda_volt_run)
    tda_temp_run = np.copy(dda_volt_run)
    for d in range(4):
        dda_volt_run[:,d] = np.histogram(dda_volt_evt[:,d], bins = bins)[0].astype(int)
        dda_curr_run[:,d] = np.histogram(dda_curr_evt[:,d], bins = bins)[0].astype(int)
        dda_temp_run[:,d] = np.histogram(dda_temp_evt[:,d], bins = temp_bins)[0].astype(int)
        tda_volt_run[:,d] = np.histogram(tda_volt_evt[:,d], bins = bins)[0].astype(int)
        tda_curr_run[:,d] = np.histogram(tda_curr_evt[:,d], bins = bins)[0].astype(int)
        tda_temp_run[:,d] = np.histogram(tda_temp_evt[:,d], bins = temp_bins)[0].astype(int)

    dda_volt.append(dda_volt_run)
    dda_curr.append(dda_curr_run)
    dda_temp.append(dda_temp_run)
    tda_volt.append(tda_volt_run)
    tda_curr.append(tda_curr_run)
    tda_temp.append(tda_temp_run)
    del hf, dda_volt_evt, dda_curr_evt, dda_temp_evt, tda_volt_evt, tda_curr_evt, tda_temp_evt 
del bad_runs, d_list, d_run_tot, bins, bin_center

run_arr = np.asarray(run_arr)
atri_volt = np.asarray(atri_volt, dtype = int)
atri_curr = np.asarray(atri_curr, dtype = int)
dda_volt = np.asarray(dda_volt, dtype = int)
dda_curr = np.asarray(dda_curr, dtype = int)
dda_temp = np.asarray(dda_temp, dtype = int)
tda_volt = np.asarray(tda_volt, dtype = int)
tda_curr = np.asarray(tda_curr, dtype = int)
tda_temp = np.asarray(tda_temp, dtype = int)
print(run_arr.shape)
print(bin_range.shape)
print(atri_volt.shape)
print(atri_curr.shape)
print(dda_volt.shape)
print(dda_curr.shape)
print(dda_temp.shape)
print(tda_volt.shape)
print(tda_curr.shape)
print(tda_temp.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Sensor_Info_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('bin_range', data=bin_range, compression="gzip", compression_opts=9)
hf.create_dataset('atri_volt', data=atri_volt, compression="gzip", compression_opts=9)
hf.create_dataset('atri_curr', data=atri_curr, compression="gzip", compression_opts=9)
hf.create_dataset('dda_volt', data=dda_volt, compression="gzip", compression_opts=9)
hf.create_dataset('dda_curr', data=dda_curr, compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp', data=dda_temp, compression="gzip", compression_opts=9)
hf.create_dataset('tda_volt', data=tda_volt, compression="gzip", compression_opts=9)
hf.create_dataset('tda_curr', data=tda_curr, compression="gzip", compression_opts=9)
hf.create_dataset('tda_temp', data=tda_temp, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







