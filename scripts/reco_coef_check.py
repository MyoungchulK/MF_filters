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

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
print(bad_runs)
print(f'# of bad runs: {len(bad_runs)}')

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    bad_idx = d_run_tot[r] in bad_runs
    #    print('bad run:', d_list[r], d_run_tot[r])
    #    continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    if g_idx != 8:
        continue

    hf = h5py.File(d_list[r], 'r')
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    coef = hf['coef'][:] # pol, rad, sol, evt
    evt = hf['evt_num'][:]
    del hf, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:]
    cut = np.in1d(evt, evt_full[qual != 0])
    del q_name, hf_q, qual, evt_full

    coef_c = np.copy(coef)
    coef_c[:,:,:,cut] = np.nan
    del cut

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    if bad_idx:
                        continue
                    if t == 0 and pol == 0 and rad == 0 and sol == 0:
                        coef_val = coef_c[pol, rad, sol][t_list[t]] > 0.125
                        if np.any(coef_val):
                            evtss = evt[t_list[t]][coef_val]
                            print(Station, d_run_tot[r], len(evtss), evtss)

    del coef, coef_c, t_list, rf_t, cal_t, soft_t

print('done!')





