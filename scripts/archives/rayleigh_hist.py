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
from tools.fft import psd_maker

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


# info data
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl/*'
d_list_chaos = glob(d_path)
d_len = len(d_list_chaos)
print(d_len)
run_tot=np.full((d_len),np.nan,dtype=int)
aa = 0
for d in d_list_chaos:
    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1
del aa, d_path

run_index = np.argsort(run_tot)
run_tot = run_tot[run_index]
d_list = []
for d in range(d_len):
    d_list.append(d_list_chaos[run_index[d]])
print(run_tot)
del d_list_chaos, d_len

# detector config
ant_num = 16
trig_type = 3


run_num = run_tot[-1] - run_tot[0] +1

# run bin
run_bins = np.linspace(0, run_num, run_num+1)
run_bin_center = (run_bins[1:] + run_bins[:-1]) * 0.5

print(run_num)
print(len(run_bin_center))

run_range = np.arange(run_tot[0],run_tot[-1]+1)
print(len(run_range))

# freq bin
freq = np.fft.rfftfreq(1216*2,0.5)

freq_bins = np.linspace(0,len(freq),len(freq) + 1)
#print(freq_bins)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) * 0.5
freq_bin_center = freq_bin_center
#print(freq_bin_center)

# rayl 2d
rayl_2d_run_w_sc = np.full((len(freq_bin_center), len(run_bin_center), ant_num),np.nan)
rayl_2d_run_wo_sc = np.full((len(freq_bin_center), len(run_bin_center), ant_num),np.nan)
print(rayl_2d_run_w_sc.shape)

mag = np.arange(-150,-100)

mag_bins = np.linspace(0,500, 500 +1)
mag_bin_center = (mag_bins[1:] + mag_bins[:-1]) * 0.5

rayl_2d_freq_w_sc = np.zeros((len(freq_bin_center), len(mag_bin_center), ant_num))
rayl_2d_freq_wo_sc = np.zeros((len(freq_bin_center), len(mag_bin_center), ant_num))

for r in tqdm(range(len(run_tot))):
    rr = np.where(run_range == run_tot[r])[0][0]

    if run_tot[r] in bad_runs:
        print('bad run:',d_list[r],run_tot[r])
        continue
    else:

        hf = h5py.File(d_list[r], 'r')

        psd_w_sc = hf['psd'][:]
        mu_wo_sc = hf['mu_wo_sc'][:]

        psd_wo_sc = psd_maker(mu_wo_sc/2, oneside = True, symmetry = True, dbm_per_hz = True)

        for a in range(ant_num):

            rayl_2d_run_w_sc[:,rr,a] = psd_w_sc[:,a]
            rayl_2d_run_wo_sc[:,rr,a] = psd_wo_sc[:,a]

            rayl_2d_freq_w_sc[:,:,a] += np.histogram2d(freq, psd_w_sc[:,a], bins = (freq_bins,mag_bins))[0]
            rayl_2d_freq_wo_sc[:,:,a] += np.histogram2d(freq, psd_wo_sc[:,a], bins = (freq_bins,mag_bins))[0]

        del hf, psd_w_sc, mu_wo_sc, psd_wo_sc

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Rayl_Hist_2d_Test_A{Station}.h5', 'w')
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)

hf.create_dataset('run_bins', data=run_bins, compression="gzip", compression_opts=9)
hf.create_dataset('run_bin_center', data=run_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('mag', data=mag, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bins', data=mag_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bin_center', data=mag_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('rayl_2d_run_w_sc', data=rayl_2d_run_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d_run_wo_sc', data=rayl_2d_run_wo_sc, compression="gzip", compression_opts=9)

hf.create_dataset('rayl_2d_freq_w_sc', data=rayl_2d_freq_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d_freq_wo_sc', data=rayl_2d_freq_wo_sc, compression="gzip", compression_opts=9)

hf.close()

print('Done!!')




