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

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/'
c_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/'

snr_bins = np.linspace(0,150,150+1)
snr_bin_center = (snr_bins[1:] + snr_bins[:-1]) / 2
snr_bin_len = len(snr_bin_center)

coef_bins = np.linspace(0,10,100+1)
coef_bin_center = (coef_bins[1:] + coef_bins[:-1]) / 2
coef_bin_len = len(coef_bin_center)

coef_tot = np.full((snr_bin_len, coef_bin_len), 0, dtype = int)
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
        coef_r = hf['coef'][0,1]
    except KeyError:
        continue
    coef_rf_r = coef_r[trig == 0]
    coef_cal_r = coef_r[trig == 1]
    coef_soft_r = coef_r[trig == 2]

    hf_s = h5py.File(f'{s_path}snr_A{Station}_R{d_run_tot[r]}.h5', 'r')
    snr_r = np.sort(hf_s['snr'][:8], axis = 0)[-2]
    snr_rf_r = snr_r[trig == 0]
    snr_cal_r = snr_r[trig == 1]
    snr_soft_r = snr_r[trig == 2]
    
    coef_tot += np.histogram2d(snr_r, coef_r, bins = (snr_bins, coef_bins))[0].astype(int)
    coef_rf += np.histogram2d(snr_rf_r, coef_rf_r, bins = (snr_bins, coef_bins))[0].astype(int)
    coef_cal += np.histogram2d(snr_cal_r, coef_cal_r, bins = (snr_bins, coef_bins))[0].astype(int)
    coef_soft += np.histogram2d(snr_soft_r, coef_soft_r,  bins = (snr_bins, coef_bins))[0].astype(int)
    del snr_rf_r, snr_cal_r, snr_soft_r

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf_q = h5py.File(f'{q_path}qual_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cuts = hf_q['tot_qual_cut_sum'][:]
    del hf_q

    snr_rf_clean = snr_r[(trig == 0) & (cuts == 0)]
    coef_rf_clean = coef_r[(trig == 0) & (cuts == 0)]
    coef_rf_cut += np.histogram2d(snr_rf_clean, coef_rf_clean, bins = (snr_bins, coef_bins))[0].astype(int)

    hf_c = h5py.File(f'{c_path}cw_cut_A{Station}_R{d_run_tot[r]}.h5', 'r')
    cw_cut = hf_c['cw_qual_cut_sum'][:]
    cw_cut += cuts   
 
    snr_cw = snr_r[(trig == 0) & (cw_cut == 0)]
    coef_cw = coef_r[(trig == 0) & (cw_cut == 0)]
    coef_rf_cw_cut += np.histogram2d(snr_cw, coef_cw, bins = (snr_bins, coef_bins))[0].astype(int)
    del snr_cw
    del hf, trig, snr_r, snr_rf_clean, cuts, hf_c, cw_cut


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('snr_bins', data=snr_bins, compression="gzip", compression_opts=9)
hf.create_dataset('snr_bin_center', data=snr_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('coef_bins', data=coef_bins, compression="gzip", compression_opts=9)
hf.create_dataset('coef_bin_center', data=coef_bin_center, compression="gzip", compression_opts=9)
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






