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

# sort
d_path_0125 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
d_list_0125, d_run_tot_0125, d_run_range = file_sorter(d_path_0125)
d_path_025 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
d_list_025, d_run_tot_025, d_run_range = file_sorter(d_path_025)
d_path_04 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_04/*'
d_list_04, d_run_tot, d_run_range = file_sorter(d_path_04)

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
del hf

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

#radius
ara_R2 = st_r**2
mwx_R2 = r_mwx**2
two_Rs = 2*r_mwx*st_r
tri_term = np.cos(st_lat) * np.cos(lat_mwx) * np.cos(st_long - lon_mwx) + np.sin(st_lat) * np.sin(lat_mwx)
radius = np.sqrt(ara_R2 + mwx_R2 - two_Rs * tri_term)
ff_mwx = interp1d(unix_mwx, radius)
del ara_R2, mwx_R2, two_Rs, tri_term

rf_tot = np.array([0], dtype = int)
flag_tot = np.full((16, 4), 0, dtype = int)
mwx_rf_tot = np.array([0], dtype = int)
r_flag_tot = np.array([0], dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    cut_tot = np.full((16, 3), np.nan, dtype = float)
    cut_tot[:,0] = cw_arr_04[:,g_idx]
    cut_tot[:,1] = cw_arr_025[:,g_idx]
    cut_tot[:,2] = cw_arr_0125[:,g_idx]

    run_list = [d_list_04[r], d_list_025[r], d_list_0125[r]]
    for h in range(3):
        hf = h5py.File(run_list[h], 'r')
        if h == 0:
            trig = hf['trig_type'][:]
            rf_count = np.count_nonzero(trig == 0)
            rf_tot[0] += rf_count
            unix_time = hf['clean_unix'][:]
            f_tot = np.full((16, len(unix_time)), 0, dtype = int)
            del trig

        ratio = np.nanmax(hf['sub_ratio'][:], axis = 0)
        r_flag = ratio > cut_tot[:,h][:, np.newaxis]        
        r_count = np.count_nonzero(r_flag, axis = 0)
        for a in range(16):
            num_c = r_count > a
            f_tot[a] += num_c
            flag_tot[a,h] += np.count_nonzero(num_c)
        if h == 2:
            flag_tot[:, 3] += np.count_nonzero(f_tot, axis = 1)
        del hf, ratio, r_flag, r_count
    del run_list, cut_tot

    if g_idx < 5:
        continue
    mwx_rf_tot[0] += rf_count
    unix_idx = np.in1d(unix_time, unix_mwx)
    if np.count_nonzero(unix_idx) == 0:
        continue
    unix_cut = unix_time[unix_idx]
    r_cut = ff_mwx(unix_cut)
    r_flag = r_cut < 22000 
    r_flag_tot[0] += np.count_nonzero(r_flag)    
    del unix_idx, unix_cut, r_cut, r_flag

print(rf_tot)
print(flag_tot)
print(mwx_rf_tot)
print(r_flag_tot)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Livetime_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('rf_tot', data=rf_tot, compression="gzip", compression_opts=9)
hf.create_dataset('flag_tot', data=flag_tot, compression="gzip", compression_opts=9)
hf.create_dataset('r_flag_tot', data=r_flag_tot, compression="gzip", compression_opts=9)
hf.create_dataset('mwx_rf_tot', data=mwx_rf_tot, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






