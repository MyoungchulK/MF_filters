import numpy as np
import os, sys
import h5py
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.interferometers import cross_correlation
from tools.interferometers import coval_sampling
from tools.antenna import antenna_combination_maker
from tools.antenna import com_dt_arr_table_maker

dpath = '/data/user/mkim/E-1_Aeff/'

dname = 'AraOut.setup_random_SN_E-1_N100_R12km_rayl_crop.txt.run'

evts = 100
runs = 1

wf_all = np.full((2048,16,evts),np.nan)
wf_len_all = np.full((16,evts),np.nan)

for a in tqdm(range(runs)):

    dat_name = dpath+dname+str(a)+'.h5'

    hf = h5py.File(dat_name, 'r')
    wf_all_indi = hf['volt'][:]/1e3
    wf_all[:,:,a*100:a*100+100] = wf_all_indi
    wf_len_all_indi = np.full((wf_all_indi.shape[1],wf_all_indi.shape[2]),np.nan)
    for evt in range(wf_all_indi.shape[2]):
        for ant in range(wf_all_indi.shape[1]):
            wf_len_all_indi[ant,evt] = np.count_nonzero(wf_all_indi[:,ant,evt])

    wf_len_all[:,a*100:a*100+100] = wf_len_all_indi
    del hf, wf_len_all_indi, wf_all_indi

print(wf_all.shape)
print(wf_len_all.shape)

spath = '/data/user/mkim/MF_Signal_Aeff/'
sname = 'AraOut.setup_random_SN_E-1_N100_R12km_rayl_crop.txt.run_tot.h5'

hf = h5py.File(spath+sname, 'r')
snr_max = hf['snr_max'][:]
del hf
print(snr_max.shape)

corr_max = np.full((2,evts),np.nan)
ant_pairs, v_pairs, h_pairs = antenna_combination_maker()
v_pairs = len(v_pairs)
print(v_pairs)
corr_map = np.full((2,180,360,evts),np.nan)

arr_table_name = '/home/mkim/analysis/MF_filters/table/Table_A2_R41.h5'
table, arr_re_table, arr1, arr2, theta, phi = com_dt_arr_table_maker(arr_table_name, ant_pairs)
print(table.shape)
print(arr1.shape)
print(arr2.shape)
del arr_re_table, theta, phi

corr, lags = cross_correlation(wf_all, wf_len_all)
print(corr.shape)
print(lags.shape)

for e in tqdm(range(evts)):
   
    snr_prod = antenna_combination_maker(arr = snr_max[:,e])[0]
    snr_prod = np.prod(snr_prod, axis = 1)
    print(snr_prod)

    # snr scaling
    v_scaling = np.nansum(snr_prod[:v_pairs])
    h_scaling = np.nansum(snr_prod[v_pairs:])

    coval = coval_sampling(corr[:,:,e], lags, table, arr1, arr2)
    coval *= snr_prod[np.newaxis, np.newaxis, :]
    del snr_prod

    corr_v = np.nansum(coval[:,:,:v_pairs],axis=2)
    corr_v *= (1./v_scaling)
    corr_h = np.nansum(coval[:,:,v_pairs:],axis=2)
    corr_h *= (1./h_scaling)
    del coval, v_scaling, h_scaling

    corr_map[0,:,:,e] = corr_v
    corr_map[1,:,:,e] = corr_h

    corr_max[0,e] = np.nanmax(corr_v)
    corr_max[1,e] = np.nanmax(corr_h)
    del corr_v, corr_h
   
print(corr_max) 

# create output dir
Output = '/home/mkim/corr/'
if not os.path.exists(Output):
    os.makedirs(Output)
os.chdir(Output)
h5_file_name=f'Corr_Signal.h5'
hf = h5py.File(h5_file_name, 'w')
hf.create_dataset('corr_max', data=corr_max, compression="gzip", compression_opts=9)
hf.create_dataset('corr_map', data=corr_map, compression="gzip", compression_opts=9)
hf.close()

print(f'output is {Output}{h5_file_name}')
print('done!')





















