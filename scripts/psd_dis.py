import numpy as np
import os, sys
import re
from glob import glob
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import config_checker
from tools.run import bad_run
from tools.run import bad_surface_run

Station = int(sys.argv[1])
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

config_count = np.zeros((6))
psd_config0 = np.zeros((2048,16,1))
psd_config1 = np.zeros((2048,16,1))
psd_config2 = np.zeros((2048,16,1))
psd_config3 = np.zeros((2048,16,1))
psd_config4 = np.zeros((2048,16,1))
psd_config5 = np.zeros((2048,16,1))
freq = np.fft.fftfreq(2048,0.5/1e9)
#print(freq[:len(freq)//2]/1e9)


d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/PSD/*'

d_list = sorted(glob(d_path))

aa = 0
cc = 0
for d in d_list:

    d_run = int(re.sub("\D", "", d[-8:-1]))
    
    try:
        a = np.where(bad_runs == d_run)[0][0]
        aa +=1
    except IndexError:
        #config_count[config_checker(Station, d_run)] += 1

        h5_file = h5py.File(d, 'r')
        Config = int(h5_file['Config'][:][0])
        config_count[Config] += 1
        
        soft_psd = h5_file['soft_psd'][:]
        soft_psd = np.repeat(soft_psd[:,:,np.newaxis],1,axis=2)
        if Config == 0:
            psd_config0 = np.append(psd_config0, soft_psd, axis=2)
        if Config == 1:
            psd_config1 = np.append(psd_config1, soft_psd, axis=2)
        if Config == 2:
            psd_config2 = np.append(psd_config2, soft_psd, axis=2)
        if Config == 3:
            psd_config3 = np.append(psd_config3, soft_psd, axis=2)
        if Config == 4:
            psd_config4 = np.append(psd_config4, soft_psd, axis=2)
        if Config == 5:
            psd_config5 = np.append(psd_config5, soft_psd, axis=2)
 
        del h5_file, Config, soft_psd

        print(cc)
        cc += 1

    #if cc == 10:
    #    break

config_count = config_count.astype(int)
print(config_count)
print(aa)

psd_0_mean = np.nanmean(psd_config0[:,:,1:], axis=2)
psd_1_mean = np.nanmean(psd_config1[:,:,1:], axis=2)
psd_2_mean = np.nanmean(psd_config2[:,:,1:], axis=2)
psd_3_mean = np.nanmean(psd_config3[:,:,1:], axis=2)
psd_4_mean = np.nanmean(psd_config4[:,:,1:], axis=2)
psd_5_mean = np.nanmean(psd_config5[:,:,1:], axis=2)
print(psd_5_mean.shape)

psd_0_std = np.nanstd(psd_config0[:,:,1:], axis=2)
psd_1_std = np.nanstd(psd_config1[:,:,1:], axis=2)
psd_2_std = np.nanstd(psd_config2[:,:,1:], axis=2)
psd_3_std = np.nanstd(psd_config3[:,:,1:], axis=2)
psd_4_std = np.nanstd(psd_config4[:,:,1:], axis=2)
psd_5_std = np.nanstd(psd_config5[:,:,1:], axis=2)
print(psd_5_std.shape)

path = '/home/mkim/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'psd_dis_A{Station}.h5', 'w')
hf.create_dataset('config_count', data=config_count, compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('psd_0_mean', data=psd_0_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_1_mean', data=psd_1_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_2_mean', data=psd_2_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_3_mean', data=psd_3_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_4_mean', data=psd_4_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_5_mean', data=psd_5_mean, compression="gzip", compression_opts=9)
hf.create_dataset('psd_0_std', data=psd_0_std, compression="gzip", compression_opts=9)
hf.create_dataset('psd_1_std', data=psd_1_std, compression="gzip", compression_opts=9)
hf.create_dataset('psd_2_std', data=psd_2_std, compression="gzip", compression_opts=9)
hf.create_dataset('psd_3_std', data=psd_3_std, compression="gzip", compression_opts=9)
hf.create_dataset('psd_4_std', data=psd_4_std, compression="gzip", compression_opts=9)
hf.create_dataset('psd_5_std', data=psd_5_std, compression="gzip", compression_opts=9)
hf.close()

print('Done!!')








