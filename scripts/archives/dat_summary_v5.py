import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/'
s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr/'
sb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_banila/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

pol_range = np.arange(3, dtype = int)
ang_range = np.arange(2, dtype = int)

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
livetime = np.full((d_len, 3), 0, dtype = float)

run_ep = np.full((0), 0, dtype = int)
evt_ep = np.copy(run_ep)
trig_ep = np.copy(run_ep)
con_ep = np.copy(run_ep)
qual_ep = np.copy(run_ep)
qual_no_s_ep = np.copy(run_ep)

coef = np.full((3, 3, 2, 0), 0, dtype = float) # pols, rads, rays, evts
coord = np.full((3, 2, 3, 2, 0), 0, dtype = float) # pols, thephi, rads, rays, evts
coef_max = np.full((3, 0), 0, dtype = float) # pols, evts
coord_max = np.full((2, 3, 0), 0, dtype = float) # thephi, pols, evts
mf_max = np.copy(coef_max) # pols, evts
mf_temp = np.full((3, 2, 0), 0, dtype = float) # pols, thephi, evts 
snr_3rd = np.copy(coef_max) # pols, evts
snr_b_3rd = np.copy(coef_max)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs[r] = hf['config'][2]
    evt = hf['evt_num'][:]
    trig_type = hf['trig_type'][:]
    num_evts = len(evt)
    evt_range = np.arange(num_evts, dtype = int)
    run_r = np.full((num_evts), d_run_tot[r], dtype = int)
    con_r = np.full((num_evts), configs[r], dtype = int)
    run_ep = np.concatenate((run_ep, run_r))
    evt_ep = np.concatenate((evt_ep, evt))
    trig_ep = np.concatenate((trig_ep, trig_type))
    con_ep = np.concatenate((con_ep, con_r))
    del trig_type, num_evts, run_r, con_r 

    coef_r = hf['coef'][:]
    coord_r = hf['coord'][:]
    coef = np.concatenate((coef, coef_r), axis = 3) 
    coord = np.concatenate((coord, coord_r), axis = 4) 
    coef_re = np.reshape(coef_r, (3, 6, -1))
    coord_re = np.reshape(coord_r, (3, 2, 6, -1))
    coord_re = np.transpose(coord_re, (1, 0, 2, 3))
    coef_max_idx = np.argmax(coef_re, axis = 1)
    coef_max_r = coef_re[pol_range[:, np.newaxis], coef_max_idx, evt_range[np.newaxis, :]]
    coord_max_r = coord_re[ang_range[:, np.newaxis, np.newaxis], pol_range[np.newaxis, :, np.newaxis], coef_max_idx, evt_range[np.newaxis, np.newaxis, :]]
    coef_max = np.concatenate((coef_max, coef_max_r), axis = 1)
    coord_max = np.concatenate((coord_max, coord_max_r), axis = 2)
    del coef_r, coord_r, hf,evt_range, coef_re, coord_re, coef_max_idx, coef_max_r, coord_max_r

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    qual_indi = hf_q['tot_qual_cut'][:]
    qual_indi[:, 14] = 0 # no l1 cut
    qual_indi[:, 15] = 0 # no rf/cal cut
    qual_indi[:, -3] = 0 # corr surface cut
    qual_indi[:, -2] = 0 # mf surface cut
    qual_indi_sum = np.nansum(qual_indi, axis = 1) != 0
    cut = np.in1d(evt, evt_full[qual])
    cut_no_s = np.in1d(evt, evt_full[qual_indi_sum])
    qual_ep = np.concatenate((qual_ep, cut.astype(int))) 
    qual_no_s_ep = np.concatenate((qual_no_s_ep, cut_no_s.astype(int))) 
    tot_live = np.nansum(hf_q['tot_qual_live_time'][:])
    bad_live = np.nansum(hf_q['tot_qual_sum_bad_live_time'][:])
    good_live = tot_live - bad_live
    livetime[r, 0] = tot_live
    livetime[r, 1] = good_live
    livetime[r, 2] = bad_live
    del q_name, hf_q, qual, evt_full, tot_live, bad_live, good_live, evt, cut, qual_indi, qual_indi_sum, cut_no_s

    bad_ant = known_issue.get_bad_antenna(d_run_tot[r])

    s_name = f'{s_path}snr_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(s_name, 'r')
    snr = hf['snr'][:]
    snr[bad_ant] = np.nan
    snr_max = np.full((3, len(snr[0])), np.nan, dtype = float)
    snr_max[0] = -np.sort(-snr[:8], axis = 0)[2]
    snr_max[1] = -np.sort(-snr[8:], axis = 0)[2]
    snr_max[2] = -np.sort(-snr, axis = 0)[2]
    snr_3rd = np.concatenate((snr_3rd, snr_max), axis = 1)
    del s_name, hf, snr, snr_max 

    s_name = f'{sb_path}snr_banila_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(s_name, 'r')
    snr = hf['snr'][:]
    snr[bad_ant] = np.nan
    snr_max = np.full((3, len(snr[0])), np.nan, dtype = float)
    snr_max[0] = -np.sort(-snr[:8], axis = 0)[2]
    snr_max[1] = -np.sort(-snr[8:], axis = 0)[2]
    snr_max[2] = -np.sort(-snr, axis = 0)[2]
    snr_b_3rd = np.concatenate((snr_b_3rd, snr_max), axis = 1)
    del bad_ant, s_name, hf, snr, snr_max

    m_name = f'{m_path}mf_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(m_name, 'r')
    mf_m = hf['mf_max'][:]
    mf_t_p = hf['mf_temp'][:, 1:3]
    mf_t_c = hf['mf_temp_com'][1:3] # of pols, theta n phi, # of evts
    mf_t = np.full((3, 2, mf_m.shape[-1]), np.nan, dtype = float)
    mf_t[:2] = mf_t_p
    mf_t[2] = mf_t_c
    mf_max = np.concatenate((mf_max, mf_m), axis = 1)
    mf_temp = np.concatenate((mf_temp, mf_t), axis = 2) 
    del m_name, hf, mf_m, mf_t, mf_t_p, mf_t_c
    
print(coef_max.shape)
print(coord_max.shape)
print(mf_max.shape)
print(mf_temp.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_no_s_ep', data=qual_no_s_ep, compression="gzip", compression_opts=9)
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.create_dataset('snr_3rd', data=snr_3rd, compression="gzip", compression_opts=9)
hf.create_dataset('snr_b_3rd', data=snr_b_3rd, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






