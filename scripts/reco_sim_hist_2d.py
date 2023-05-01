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
#from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import get_path_info_v2

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f
sim_type = str(sys.argv[4])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_sim/*{sim_type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

if sim_type == 'signal':
    q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/One_Weight_Pad_mass_A{Station}.h5'
    hf = h5py.File(q_path, 'r')
    prob = hf['probability'][:]
    evt_rate = hf['evt_rate'][:]
    evt_rate[prob >= 1] = 0
    print(prob.shape)
    print(evt_rate.shape)
    fla_w = hf['flavor'][:]
    en_w = hf['exponent'][:, 0]
    con_w = hf['config'][:]
    run_w = hf['sim_run'][:]
    print(fla_w.shape)
    print(en_w.shape)
    print(con_w.shape)
    print(run_w.shape)
    del q_path, hf, prob

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)

nan_counts = np.full((d_len), 0, dtype = int)
config = np.full((d_len), 0, dtype = int)
sim_run = np.copy(config)
flavor = np.copy(config)
en = np.copy(config)

#map_az = np.full((d_len, 2, a_bin_len, z_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
#map_az_w = np.full((d_len, 2, a_bin_len, z_bin_len), 0, dtype = float)
#map_ac = np.full((d_len, 2, a_bin_len, c_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
#map_ac_w = np.full((d_len, 2, a_bin_len, c_bin_len), 0, dtype = float)
#map_zc = np.full((d_len, 2, z_bin_len, c_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
#map_zc_w = np.full((d_len, 2, z_bin_len, c_bin_len), 0, dtype = float)
map_az = np.full((num_configs, 3, 2, a_bin_len, z_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_az_w = np.full((num_configs, 3, 2, a_bin_len, z_bin_len), 0, dtype = float)
map_ac = np.full((num_configs, 3, 2, a_bin_len, c_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_ac_w = np.full((num_configs, 3, 2, a_bin_len, c_bin_len), 0, dtype = float)
map_zc = np.full((num_configs, 3, 2, z_bin_len, c_bin_len), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_zc_w = np.full((num_configs, 3, 2, z_bin_len, c_bin_len), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    entry_num = hf['entry_num'][:]
    num_evts = len(entry_num)
    con = hf['config'][:]
    sim_run[r] = int(con[1])
    config[r] = int(con[2])
    if sim_type == 'signal':
        flavor[r] = int(con[4])
        en[r] = int(get_path_info_v2(d_list[r], '_E', '_F')) - 9
        w_idx = np.all((fla_w == flavor[r], en_w == en[r], con_w == config[r], run_w == sim_run[r]), axis = 0)
        evt_w = evt_rate[w_idx][0]
    else:
        evt_w = np.full((num_evts), 1, dtype = float)

    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:] # pol, rad, sol, evt
    coef_re = np.reshape(coef, (2, 4, -1))
    coef_max = np.nanmax(coef_re, axis = 1)
    coord_re = np.reshape(coord, (2, 2, 4, -1))
    coord_max = np.full((2, 2, num_evts), np.nan, dtype = float)
    counts = 0
    for e in range(num_evts):    
        try:
            coef_max_idx = np.nanargmax(coef_re[:, :, e], axis = 1)
        except ValueError:
            counts += 1
            continue
        coord_max[0, :, e] = coord_re[0, :, coef_max_idx[0], e]
        coord_max[1, :, e] = coord_re[1, :, coef_max_idx[1], e]
        del coef_max_idx
    nan_counts[r] = counts
    del hf, coord, coef, coef_re, coord_re, num_evts, entry_num

    con_idx = int(config[r] - 1)
    if sim_type == 'signal':
        fla_idx = int(flavor[r] - 1)
    else:
        fla_idx = 0

    for pol in range(2):
        #map_az[r, pol] = np.histogram2d(coord_max[pol, 1], coord_max[pol, 0], bins = (a_bins, z_bins))[0].astype(int)
        #map_az_w[r, pol] = np.histogram2d(coord_max[pol, 1], coord_max[pol, 0], bins = (a_bins, z_bins), weights = evt_w)[0]
        #map_ac[r, pol] = np.histogram2d(coord_max[pol, 1], coef_max[pol], bins = (a_bins, c_bins))[0].astype(int)
        #map_ac_w[r, pol] = np.histogram2d(coord_max[pol, 1], coef_max[pol], bins = (a_bins, c_bins), weights = evt_w)[0]
        #map_zc[r, pol] = np.histogram2d(coord_max[pol, 0], coef_max[pol], bins = (z_bins, c_bins))[0].astype(int)
        #map_zc_w[r, pol] = np.histogram2d(coord_max[pol, 0], coef_max[pol], bins = (z_bins, c_bins), weights = evt_w)[0]
        map_az[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 1], coord_max[pol, 0], bins = (a_bins, z_bins))[0].astype(int)
        map_az_w[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 1], coord_max[pol, 0], bins = (a_bins, z_bins), weights = evt_w)[0]
        map_ac[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 1], coef_max[pol], bins = (a_bins, c_bins))[0].astype(int)
        map_ac_w[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 1], coef_max[pol], bins = (a_bins, c_bins), weights = evt_w)[0]
        map_zc[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 0], coef_max[pol], bins = (z_bins, c_bins))[0].astype(int)
        map_zc_w[con_idx, fla_idx, pol] += np.histogram2d(coord_max[pol, 0], coef_max[pol], bins = (z_bins, c_bins), weights = evt_w)[0]
    del con, evt_w, coef_max, coord_max

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Sim_Map_{sim_type}_2d_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('en', data=en, compression="gzip", compression_opts=9)
hf.create_dataset('nan_counts', data=nan_counts, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_az', data=map_az, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_w', data=map_az_w, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac', data=map_ac, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_w', data=map_ac_w, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc', data=map_zc, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_w', data=map_zc_w, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






