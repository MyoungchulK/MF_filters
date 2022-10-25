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

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

z_bins = np.linspace(0, 180, 180 + 1, dtype = int)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1, dtype = int)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)

map_tot = np.full((a_bin_len, z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # trig, pol, rad, sol
map_daq = np.copy(map_tot)
map_fir = np.copy(map_tot)
map_dda = np.copy(map_tot)
map_rate = np.copy(map_tot)
map_l1 = np.copy(map_tot)
map_short = np.copy(map_tot)
map_bad = np.copy(map_tot)
map_log = np.copy(map_tot)
map_wb = np.copy(map_tot)
map_pole = np.copy(map_tot)
map_un = np.copy(map_tot)
map_ped = np.copy(map_tot)
map_cut = np.copy(map_tot)

daq_idx = np.array([0,1,2,3,4,5,6,7,8], dtype = int)
fir_idx =  np.array([9], dtype = int)
dda_idx = np.array([10,11], dtype = int)
rate_idx = np.array([12,13], dtype = int)
l1_idx =  np.array([15], dtype = int)
short_idx =  np.array([16], dtype = int)
bad_idx = np.array([17,18], dtype = int)
log_idx =  np.array([19], dtype = int)
wb_idx =  np.array([20], dtype = int)
pole_idx =  np.array([21], dtype = int)
un_idx =  np.array([22], dtype = int)
ped_idx =  np.array([26], dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    coord = hf['coord'][:]
    evt = hf['evt_num'][:]
    del hf, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut'][:]
    qual[:, 14] = 0
    daq = np.in1d(evt, evt_full[np.nansum(qual[:, daq_idx], axis = 1) != 0])
    fir = np.in1d(evt, evt_full[np.nansum(qual[:, fir_idx], axis = 1) != 0])
    dda = np.in1d(evt, evt_full[np.nansum(qual[:, dda_idx], axis = 1) != 0])
    rate = np.in1d(evt, evt_full[np.nansum(qual[:, rate_idx], axis = 1) != 0])
    l1 = np.in1d(evt, evt_full[np.nansum(qual[:, l1_idx], axis = 1) != 0])
    short = np.in1d(evt, evt_full[np.nansum(qual[:, short_idx], axis = 1) != 0])
    bad = np.in1d(evt, evt_full[np.nansum(qual[:, bad_idx], axis = 1) != 0])
    log = np.in1d(evt, evt_full[np.nansum(qual[:, log_idx], axis = 1) != 0])
    wb = np.in1d(evt, evt_full[np.nansum(qual[:, wb_idx], axis = 1) != 0])
    pole = np.in1d(evt, evt_full[np.nansum(qual[:, pole_idx], axis = 1) != 0])
    un = np.in1d(evt, evt_full[np.nansum(qual[:, un_idx], axis = 1) != 0])
    ped = np.in1d(evt, evt_full[np.nansum(qual[:, ped_idx], axis = 1) != 0])
    cut = np.in1d(evt, evt_full[np.nansum(qual, axis = 1) != 0])
    del q_name, hf_q, qual, evt_full, evt

    c_daq = np.copy(coord)
    c_daq[:,:,:,:,daq] = np.nan
    c_fir = np.copy(coord)
    c_fir[:,:,:,:,fir] = np.nan
    c_dda = np.copy(coord)
    c_dda[:,:,:,:,dda] = np.nan
    c_rate = np.copy(coord)
    c_rate[:,:,:,:,rate] = np.nan
    c_l1 = np.copy(coord)
    c_l1[:,:,:,:,l1] = np.nan
    c_short = np.copy(coord)
    c_short[:,:,:,:,short] = np.nan
    c_bad = np.copy(coord)
    c_bad[:,:,:,:,bad] = np.nan
    c_log = np.copy(coord)
    c_log[:,:,:,:,log] = np.nan
    c_wb = np.copy(coord)
    c_wb[:,:,:,:,wb] = np.nan
    c_pole = np.copy(coord)
    c_pole[:,:,:,:,pole] = np.nan
    c_un = np.copy(coord)
    c_un[:,:,:,:,un] = np.nan
    c_ped = np.copy(coord)
    c_ped[:,:,:,:,ped] = np.nan
    c_cut = np.copy(coord)
    c_cut[:,:,:,:,cut] = np.nan

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    map_tot[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coord[pol, 1, rad, sol][t_list[t]], coord[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_daq[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_daq[pol, 1, rad, sol][t_list[t]], c_daq[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_fir[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_fir[pol, 1, rad, sol][t_list[t]], c_fir[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_dda[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_dda[pol, 1, rad, sol][t_list[t]], c_dda[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_rate[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_rate[pol, 1, rad, sol][t_list[t]], c_rate[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_l1[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_l1[pol, 1, rad, sol][t_list[t]], c_l1[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_short[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_short[pol, 1, rad, sol][t_list[t]], c_short[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_bad[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_bad[pol, 1, rad, sol][t_list[t]], c_bad[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_log[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_log[pol, 1, rad, sol][t_list[t]], c_log[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_wb[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_wb[pol, 1, rad, sol][t_list[t]], c_wb[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_pole[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_pole[pol, 1, rad, sol][t_list[t]], c_pole[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_un[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_un[pol, 1, rad, sol][t_list[t]], c_un[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_ped[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_ped[pol, 1, rad, sol][t_list[t]], c_ped[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
                    map_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(c_cut[pol, 1, rad, sol][t_list[t]], c_cut[pol, 0, rad, sol][t_list[t]], bins = (a_bins, z_bins))[0].astype(int)
    del coord, t_list, rf_t, cal_t, soft_t, c_daq, c_fir, c_dda, c_rate, c_l1, c_short, c_bad, c_log, c_wb, c_pole, c_un, c_ped, c_cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_tot', data=map_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_daq', data=map_daq, compression="gzip", compression_opts=9)
hf.create_dataset('map_fir', data=map_fir, compression="gzip", compression_opts=9)
hf.create_dataset('map_dda', data=map_dda, compression="gzip", compression_opts=9)
hf.create_dataset('map_rate', data=map_rate, compression="gzip", compression_opts=9)
hf.create_dataset('map_l1', data=map_l1, compression="gzip", compression_opts=9)
hf.create_dataset('map_short', data=map_short, compression="gzip", compression_opts=9)
hf.create_dataset('map_bad', data=map_bad, compression="gzip", compression_opts=9)
hf.create_dataset('map_log', data=map_log, compression="gzip", compression_opts=9)
hf.create_dataset('map_wb', data=map_wb, compression="gzip", compression_opts=9)
hf.create_dataset('map_pole', data=map_pole, compression="gzip", compression_opts=9)
hf.create_dataset('map_un', data=map_un, compression="gzip", compression_opts=9)
hf.create_dataset('map_ped', data=map_ped, compression="gzip", compression_opts=9)
hf.create_dataset('map_cut', data=map_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






