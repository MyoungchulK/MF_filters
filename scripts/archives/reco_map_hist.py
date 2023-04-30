import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Radius = int(sys.argv[2])

if Radius == 41:
    r_idx = 0
if Radius == 300:
    r_idx = 1

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/'
c_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/'

z_bins = np.linspace(0, 180, 180 + 1, dtype = int)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1, dtype = int)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)

coef_tot = np.full((a_bin_len, z_bin_len), 0, dtype = int)
coef_rf = np.copy(coef_tot)
coef_cal = np.copy(coef_tot)
coef_soft = np.copy(coef_tot)
coef_rf_cut = np.copy(coef_tot)
coef_rf_cw_cut = np.copy(coef_tot)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    try:
        trig = hf['trig_type'][:]
        coef_r = hf['coord'][:,0,r_idx]
    except KeyError:
        continue
    coef_rf_r = coef_r[:,trig == 0]
    coef_cal_r = coef_r[:,trig == 1]
    coef_soft_r = coef_r[:,trig == 2]

    coef_tot += np.histogram2d(coef_r[1], coef_r[0], bins = (a_bins, z_bins))[0].astype(int)
    coef_rf += np.histogram2d(coef_rf_r[1], coef_rf_r[0], bins = (a_bins, z_bins))[0].astype(int)
    coef_cal += np.histogram2d(coef_cal_r[1], coef_cal_r[0], bins = (a_bins, z_bins))[0].astype(int)
    coef_soft += np.histogram2d(coef_soft_r[1], coef_soft_r[0],  bins = (a_bins, z_bins))[0].astype(int)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf_q = h5py.File(f'{q_path}qual_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cuts = hf_q['tot_qual_cut_sum'][:]
    del hf_q

    coef_rf_clean = coef_r[:,(trig == 0) & (cuts == 0)]
    coef_rf_cut += np.histogram2d(coef_rf_clean[1], coef_rf_clean[0], bins = (a_bins, z_bins))[0].astype(int)

    hf_c = h5py.File(f'{c_path}cw_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cw_cut = hf_c['cw_qual_cut_sum'][:]
    cw_cut += cuts   
 
    coef_cw = coef_r[:,(trig == 0) & (cw_cut == 0)]
    coef_rf_cw_cut += np.histogram2d(coef_cw[1], coef_cw[0], bins = (a_bins, z_bins))[0].astype(int)
    del hf, trig, cuts, hf_c, cw_cut


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_A{Station}_R{Radius}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('coef_tot', data=coef_tot, compression="gzip", compression_opts=9)
hf.create_dataset('coef_rf', data=coef_rf, compression="gzip", compression_opts=9)
hf.create_dataset('coef_cal', data=coef_cal, compression="gzip", compression_opts=9)
hf.create_dataset('coef_soft', data=coef_soft, compression="gzip", compression_opts=9)
hf.create_dataset('coef_rf_cut', data=coef_rf_cut, compression="gzip", compression_opts=9)
hf.create_dataset('coef_rf_cw_cut', data=coef_rf_cw_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






