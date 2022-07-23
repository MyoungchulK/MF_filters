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
num_ants = 16

if Station == 2:
            num_configs = 6
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 1000], dtype = float)
            cw_arr_04[:,1] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)

            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.16, 0.16, 0.16, 0.14, 0.14, 0.14, 0.14, 0.18,  0.2, 0.16, 0.24, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.16, 0.16,  0.2, 0.14, 0.12, 0.24, 0.14,  0.2,  0.2, 0.18, 0.24, 0.16, 0.18, 0.16, 1000], dtype = float)
            cw_arr_025[:,2] = np.array([0.14, 0.16, 0.14, 0.14,  0.1,  0.1, 0.14, 0.14, 0.18, 0.18, 0.18, 0.26, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.16, 0.26, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.14, 0.18, 0.14, 0.12, 0.14, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([0.12, 0.12,  0.1, 0.12,  0.1,  0.1,  0.1,  0.1, 0.14, 0.16, 0.14, 0.18, 0.12, 0.12, 0.12, 1000], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([0.06, 0.18, 0.06, 0.14, 0.12, 0.08, 0.08, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08,  0.2,  0.1, 1000], dtype = float)
            cw_arr_0125[:,1] = np.array([0.06,  0.2, 0.06, 0.18,  0.2, 0.08,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.16, 0.06, 0.12, 0.14,  0.1,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.14, 0.04, 0.08, 0.22, 0.08, 0.06, 0.06,  0.1,  0.1, 0.08, 0.08, 0.06, 0.14, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.04, 0.14, 0.04, 0.06, 0.22, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.06, 0.06, 0.14, 0.06, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.04, 0.08, 0.04, 0.06, 0.14, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.08, 0.06, 0.12, 0.08, 1000], dtype = float)

if Station == 3:
            num_configs = 7
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.06, 0.04], dtype = float)
            cw_arr_04[:,1] = np.array([0.08, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.08, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.06, 0.04, 0.06, 1000, 0.06, 0.05, 0.06, 1000, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04,  0.1, 0.04, 0.04, 0.12, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04], dtype = float)
            cw_arr_04[:,6] = np.array([1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.12, 0.02, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.08, 0.06], dtype = float)

            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.12, 0.12, 0.12, 0.16, 0.12, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16, 0.16, 0.14], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.12, 0.12, 0.14, 0.16, 0.14, 0.14, 0.16, 0.16, 0.14,  0.2, 0.14, 0.14, 0.14, 0.16, 0.16], dtype = float)
            cw_arr_025[:,2] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.12, 0.14, 1000, 0.12, 0.14, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.14, 0.14, 1000, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.14, 0.12, 0.12, 1000, 0.16, 0.16, 0.12, 1000, 0.18, 0.14, 0.16, 1000, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([ 0.1, 0.08, 0.12, 0.08,  0.1, 0.12, 0.08, 0.12, 0.12,  0.1, 0.12, 0.12,  0.1, 0.14, 0.14, 0.12], dtype = float)
            cw_arr_025[:,6] = np.array([1000, 0.06,  0.1, 0.08, 1000,  0.1, 0.06, 0.08, 1000, 0.12, 0.14,  0.1, 1000, 0.14, 0.12,  0.1], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([ 0.1, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.12,  0.1, 0.08, 0.16, 0.06,  0.1, 0.14,  0.1,  0.1], dtype = float)
            cw_arr_0125[:,1] = np.array([0.14, 0.06, 0.06, 0.06,  0.1,  0.1, 0.14, 0.12, 0.12, 0.08, 0.16, 0.08,  0.1, 0.14, 0.12,  0.1], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000,  0.1,  0.1,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000, 0.08,  0.1, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.08,  0.1, 0.08, 1000,  0.1, 0.08,  0.1, 1000,  0.1, 0.12,  0.1, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.06, 0.04, 0.06, 0.04, 0.08, 0.08, 0.04,  0.1, 0.12, 0.06, 0.08, 0.06, 0.08,  0.1,  0.1, 0.08], dtype = float)
            cw_arr_0125[:,6] = np.array([1000, 0.04, 0.06,  0.2, 1000, 0.06, 0.04, 0.18, 1000, 0.06, 0.08, 0.18, 1000,  0.1, 0.08, 0.18], dtype = float)
del num_ants, num_configs

# sort
d_path_0125 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
d_list_0125, d_run_tot_0125, d_run_range = file_sorter(d_path_0125)
d_path_025 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
d_list_025, d_run_tot_025, d_run_range = file_sorter(d_path_025)
d_path_04 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_04/*'
d_list_04, d_run_tot, d_run_range = file_sorter(d_path_04)
del d_path_0125, d_run_range, d_path_025, d_path_04

#mwx
cw_h5_path = '/home/mkim/analysis/MF_filters/data/cw_log/'
mwx_name = f'{cw_h5_path}mwx_tot_R.h5'
hf = h5py.File(mwx_name, 'r')
unix_mwx = hf['DataSrvTime_int'][:]
unix_mwx = unix_mwx.flatten()
unix_mwx = unix_mwx[~np.isnan(unix_mwx)]
unix_mwx = unix_mwx.astype(int)
lat_mwx = hf['gm_lati_int'][:]
lat_mwx = lat_mwx.flatten()
lat_mwx = lat_mwx[~np.isnan(lat_mwx)]
lon_mwx = hf['gm_lon_int'][:]
lon_mwx = lon_mwx.flatten()
lon_mwx = lon_mwx[~np.isnan(lon_mwx)]
r_mwx = hf['gm_r_int'][:]
r_mwx = r_mwx.flatten()
r_mwx = r_mwx[~np.isnan(r_mwx)]
del cw_h5_path, mwx_name, hf

#araroot
import ROOT
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libRootFftwWrapper.so.3.0.1")
from scipy.interpolate import interp1d
ara_geom = ROOT.AraGeomTool.Instance()
stationVector = ara_geom.getStationVector(Station)
st_long = ara_geom.getLongitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the longitude
st_lat = ara_geom.getGeometricLatitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the latitude
fGeoidC=6356752.3
fIceThicknessSP=2646.28
st_r = fGeoidC + fIceThicknessSP
st_long = np.radians(st_long)
st_lat = np.radians(st_lat)
del ara_geom, stationVector, fGeoidC, fIceThicknessSP 

#radius
ara_R2 = st_r**2
mwx_R2 = r_mwx**2
two_Rs = 2*r_mwx*st_r
tri_term = np.cos(st_lat) * np.cos(lat_mwx) * np.cos(st_long - lon_mwx) + np.sin(st_lat) * np.sin(lat_mwx)
radius = np.sqrt(ara_R2 + mwx_R2 - two_Rs * tri_term)
ff_mwx = interp1d(unix_mwx, radius)
del ara_R2, mwx_R2, two_Rs, tri_term, lat_mwx, lon_mwx, r_mwx, st_r, st_long, st_lat, radius 
print('radius is done!')

sec_to_min = 60

tot_time = np.full((len(d_run_tot)), np.nan, dtype = float)
tot_num_evts = np.copy(tot_time)
tot_evt_per_min = np.full((1000, len(d_run_tot)), np.nan, dtype = float)
tot_sec_per_min = np.copy(tot_evt_per_min)
tot_time_bins = np.copy(tot_evt_per_min)
tot_cut_evt_per_min_r = np.copy(tot_evt_per_min)
tot_cut_evt_per_min = np.full((3, 16, 1000, len(d_run_tot)), np.nan, dtype = float)
tot_sum_cut_evt_per_min = np.full((16, 1000, len(d_run_tot)), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
   
  #if r < 10:

    hf = h5py.File(d_list_04[r], 'r')
    unix_time = hf['unix_time'][:]
    trig_type = hf['trig_type'][:]

    rc_trig = trig_type != 2
    num_evts = np.count_nonzero(rc_trig)
    tot_num_evts[r] = num_evts
    tot_t = np.abs(unix_time[-1] - unix_time[0])
    tot_time[r] = tot_t
    del trig_type, num_evts, tot_t

    time_bins = np.arange(np.nanmin(unix_time), np.nanmax(unix_time)+1, sec_to_min, dtype = int)
    time_bins = time_bins.astype(float)
    time_bins -= 0.5
    time_bins = np.append(time_bins, np.nanmax(unix_time) + 0.5)
    tot_time_bins[:len(time_bins), r] = time_bins

    sec_per_min = np.diff(time_bins)
    tot_sec_per_min[:len(sec_per_min), r] = sec_per_min
    del sec_per_min

    rc_trig_evt = rc_trig.astype(int)
    rc_trig_evt = rc_trig_evt.astype(float)
    rc_trig_evt[rc_trig_evt < 0.5] = np.nan
    rc_trig_evt *= unix_time
    evt_per_min = np.histogram(rc_trig_evt, bins = time_bins)[0]
    tot_evt_per_min[:len(evt_per_min), r] = evt_per_min
    del rc_trig_evt, evt_per_min

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    cut_tot = np.full((16, 3), np.nan, dtype = float)
    cut_tot[:,0] = cw_arr_04[:,g_idx]
    cut_tot[:,1] = cw_arr_025[:,g_idx]
    cut_tot[:,2] = cw_arr_0125[:,g_idx]

    clean_unix = hf['clean_unix'][:]
    clean_evt = hf['clean_evt'][:]
    evt_num = hf['evt_num'][:]
    tot_cut = np.full((16, len(unix_time)), 0, dtype = int)
    run_list = [d_list_04[r], d_list_025[r], d_list_0125[r]]
    for h in range(3):
        if h == 0:
            ratio = np.nanmax(hf['sub_ratio'][:], axis = 0) # (pad, ant, evt) -> (ant,evt)
        else:
            hf1 = h5py.File(run_list[h], 'r')
            ratio = np.nanmax(hf1['sub_ratio'][:], axis = 0) # (pad, ant, evt) -> (ant,evt)
            del hf1
        r_flag = ratio > cut_tot[:,h][:, np.newaxis] # (ant,evt)
        r_count = np.count_nonzero(r_flag, axis = 0) # (evt)
        del ratio, r_flag
        for a in range(16):
            num_c = (r_count > a).astype(int)

            evt_cut = clean_evt[num_c != 0]
            tot_cut[a] += np.in1d(evt_num, evt_cut).astype(int)

            num_c = num_c.astype(float)
            num_c[num_c < 0.5] = np.nan
            num_c *= clean_unix
            cut_evt_per_min = np.histogram(num_c, bins = time_bins)[0]
            tot_cut_evt_per_min[h, a, :len(cut_evt_per_min), r] = cut_evt_per_min
            del num_c, evt_cut, cut_evt_per_min 
        del r_count
    del hf, cut_tot, clean_unix, clean_evt, evt_num, run_list 

    if g_idx > 4:
        rc_unix = unix_time[rc_trig]
        unix_idx = np.in1d(rc_unix, unix_mwx)
        if np.count_nonzero(unix_idx) != 0:
            unix_cut = rc_unix[unix_idx]
            r_cut = ff_mwx(unix_cut)
            r_flag = (r_cut < 17000).astype(int)
            r_flag = r_flag.astype(float)
            r_flag[r_flag < 0.5] = np.nan
            r_flag *= unix_cut
            del r_cut 

            r_flag_unix = r_flag[~np.isnan(r_flag)]
            r_flag_unix = r_flag_unix.astype(int)
            r_flag_unix = np.in1d(unix_time, r_flag_unix).astype(int)
            tot_cut += r_flag_unix[np.newaxis, :]
            del r_flag_unix

            cut_evt_per_min_r = np.histogram(r_flag, bins = time_bins)[0]
            tot_cut_evt_per_min_r[:len(cut_evt_per_min_r), r] = cut_evt_per_min_r
            del unix_cut, r_flag, cut_evt_per_min_r
        del rc_unix, unix_idx
    del rc_trig, g_idx

    tot_idx = tot_cut < 0.5
    tot_cut = tot_cut.astype(float)
    tot_cut[~tot_idx] = 1
    tot_cut[tot_idx] = np.nan
    tot_cut *= unix_time[np.newaxis, :]
    del tot_idx
    for a in range(16):
        sum_cut_evt_per_min = np.histogram(tot_cut[a], bins = time_bins)[0]
        tot_sum_cut_evt_per_min[a, :len(sum_cut_evt_per_min), r] = sum_cut_evt_per_min
    del unix_time, time_bins, tot_cut

del unix_mwx, ff_mwx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_cw_cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('tot_time', data=tot_time, compression="gzip", compression_opts=9)
hf.create_dataset('tot_num_evts', data=tot_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('tot_evt_per_min', data=tot_evt_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_sec_per_min', data=tot_sec_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_time_bins', data=tot_time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('tot_sum_cut_evt_per_min', data=tot_sum_cut_evt_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_evt_per_min_r', data=tot_cut_evt_per_min_r, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_evt_per_min', data=tot_cut_evt_per_min, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






