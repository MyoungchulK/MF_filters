import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
dtype = '_wb_002'

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_sim{dtype}/*'
print(d_path)
d_list = glob(d_path)

hf = h5py.File(d_list[0], 'r')
sel_entries = hf['sel_entries'][:]
bp_max = hf['bp_sky_max'][:]
cw_max = hf['bp_cw_sky_max'][:]
bp_coord = hf['bp_sky_coord'][:]
cw_coord = hf['bp_cw_sky_coord'][:]
bp_cw_num_freqs = hf['bp_cw_num_freqs'][:]
#bp_cw_num_freq_errs = hf['bp_cw_num_freq_errs'][:]
bp_cw_num_amps = hf['bp_cw_num_amps'][:]
bp_cw_num_amp_errs = hf['bp_cw_num_amp_errs'][:]
#bp_cw_num_phases = hf['bp_cw_num_phases'][:]
bp_cw_num_phase_errs = hf['bp_cw_num_phase_errs'][:]
bp_cw_num_powers = hf['bp_cw_num_powers'][:]
bp_cw_num_ratios = hf['bp_cw_num_ratios'][:]
run = np.full((len(sel_entries)), int(d_list[0][-9:-3]), dtype = int)
del hf

count = 0
for r in tqdm(d_list):

    #if r <10:
    if count == 0:
        count += 1
        continue

    hf = h5py.File(r, 'r')
    bp_r = hf['bp_sky_max'][:]
    cw_r = hf['bp_cw_sky_max'][:]
    bp_c_r = hf['bp_sky_coord'][:]
    cw_c_r = hf['bp_cw_sky_coord'][:]
    freqs = hf['bp_cw_num_freqs'][:]
    #freq_errs = hf['bp_cw_num_freq_errs'][:]
    amps = hf['bp_cw_num_amps'][:]
    amp_errs = hf['bp_cw_num_amp_errs'][:]
    #phases = hf['bp_cw_num_phases'][:]
    phase_errs = hf['bp_cw_num_phase_errs'][:]
    powers = hf['bp_cw_num_powers'][:]
    ratios = hf['bp_cw_num_ratios'][:]
    sel = hf['sel_entries'][:]
    r_r = np.full((len(sel)), int(r[-9:-3]), dtype = int)

    sel_entries = np.append(sel_entries, sel)
    run = np.append(run, r_r)
    bp_max = np.append(bp_max, bp_r, axis = 1)
    cw_max = np.append(cw_max, cw_r, axis = 1)
    bp_coord = np.append(bp_coord, bp_c_r, axis = 2)
    cw_coord = np.append(cw_coord, cw_c_r, axis = 2)
    bp_cw_num_freqs = np.append(bp_cw_num_freqs, freqs, axis = 2)
    #bp_cw_num_freq_errs = np.append(bp_cw_num_freq_errs, freq_errs, axis = 2)
    #bp_cw_num_phases = np.append(bp_cw_num_phases, phases, axis = 2)
    bp_cw_num_phase_errs = np.append(bp_cw_num_phase_errs, phase_errs, axis = 2)
    bp_cw_num_amps = np.append(bp_cw_num_amps, amps, axis = 2)
    bp_cw_num_amp_errs = np.append(bp_cw_num_amp_errs, amp_errs, axis = 2)
    bp_cw_num_powers = np.append(bp_cw_num_powers, powers, axis = 2)
    bp_cw_num_ratios = np.append(bp_cw_num_ratios, ratios, axis = 2)
    #del hf, bp_r, cw_r, bp_c_r, cw_c_r, freqs, freq_errs, amps, amp_errs, phases, phase_errs, powers, ratios, sel, r_r
    del hf, bp_r, cw_r, bp_c_r, cw_c_r, freqs, amps, amp_errs, phase_errs, powers, ratios, sel, r_r
    count += 1

print(sel_entries.shape)
print(run.shape)
print(bp_cw_num_freqs.shape)
#print(bp_cw_num_freq_errs.shape)
#print(bp_cw_num_phases.shape)
print(bp_cw_num_phase_errs.shape)
print(bp_cw_num_amps.shape)
print(bp_cw_num_amp_errs.shape)
print(bp_cw_num_powers.shape)
print(bp_cw_num_powers.shape)
print(bp_max.shape)
print(cw_max.shape)
print(bp_coord.shape)
print(cw_coord.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Sim_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('bp_max', data=bp_max, compression="gzip", compression_opts=9)
hf.create_dataset('cw_max', data=cw_max, compression="gzip", compression_opts=9)
hf.create_dataset('bp_coord', data=bp_coord, compression="gzip", compression_opts=9)
hf.create_dataset('cw_coord', data=cw_coord, compression="gzip", compression_opts=9)
hf.create_dataset('sel_entries', data=sel_entries, compression="gzip", compression_opts=9)
hf.create_dataset('run', data=run, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_freqs', data=bp_cw_num_freqs, compression="gzip", compression_opts=9)
#hf.create_dataset('bp_cw_num_freq_errs', data=bp_cw_num_freq_errs, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_amps', data=bp_cw_num_amps, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_amp_errs', data=bp_cw_num_amp_errs, compression="gzip", compression_opts=9)
#hf.create_dataset('bp_cw_num_phases', data=bp_cw_num_phases, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_phase_errs', data=bp_cw_num_phase_errs, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_powers', data=bp_cw_num_powers, compression="gzip", compression_opts=9)
hf.create_dataset('bp_cw_num_ratios', data=bp_cw_num_ratios, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






