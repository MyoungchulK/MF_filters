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
from tools.ara_known_issue import get_bad_run_list

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

rayl_bad_runs = get_bad_run_list(f'../data/rayl_runs/rayl_run_A{Station}.txt', verbose = True)
bad_runs = np.concatenate((bad_runs, rayl_bad_runs), axis = None, dtype = int)
bad_runs = np.unique(bad_runs).astype(int)
del rayl_bad_runs

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl_full/*'
d_list_old, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range, d_path

clean_idx = ~np.in1d(d_run_tot, bad_runs)
d_list = []
for d in range(d_len):
    if clean_idx[d]:
        d_list.append(d_list_old[d])
d_run_tot = d_run_tot[clean_idx]
d_len = len(d_run_tot)
print('Total Runs:', d_len)
del bad_runs, clean_idx

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
years = np.copy(configs)

hf = h5py.File(d_list[0], 'r')
f_bin_center = hf['freq_range'][:]
del hf
df = np.abs(f_bin_center[1] - f_bin_center[0])
f_bins = np.append(f_bin_center - df/2, f_bin_center[-1] + df/2)
f_bin_len = len(f_bin_center)
a_bins = np.linspace(0, 320, 640 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)
r_bin_center = np.arange(d_len, dtype = int)
r_bins = np.append(r_bin_center.astype(float) - 0.5, r_bin_center[-1].astype(float) + 0.5)
del df

rayl = np.full((d_len, f_bin_len, 16), np.nan, dtype = float)
sc = np.copy(rayl)
print(f'array dim.: {rayl.shape}')
print(f'array size: ~{np.round(rayl.nbytes/1024/1024)} MB')
del d_len, a_bin_len, f_bin_len

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    con = hf['config'][:]
    years[r] = con[3]
    configs[r] = con[2]
    del con

    rayl[r] = np.nansum(hf['soft_rayl'][:], axis = 0)
    sc[r] = hf['soft_sc'][:]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Rayl_Map_1st_blk_1d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('f_bins', data=f_bins, compression="gzip", compression_opts=9)
hf.create_dataset('f_bin_center', data=f_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('years', data=years, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('rayl', data=rayl, compression="gzip", compression_opts=9)
hf.create_dataset('sc', data=sc, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))



