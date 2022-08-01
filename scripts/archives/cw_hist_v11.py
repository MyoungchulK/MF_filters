import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
import ROOT
from scipy.interpolate import interp1d

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libRootFftwWrapper.so.3.0.1")

Station = int(sys.argv[1])
d_type = str(sys.argv[2])
#d_type = '04'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_{d_type}/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
hf = h5py.File(d_list[0], 'r')
ratio_bin_center = hf['ratio_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
ratio_len = len(ratio_bin_center)
del hf

r_bins = np.arange(0, 50001, 50)
r_bin_center = (r_bins[1:] + r_bins[:-1]) / 2
r_len = len(r_bin_center)

time_bins = np.arange(0, 60*5+1, 1)
time_bin_center = (time_bins[1:] + time_bins[:-1]) / 2
time_len = len(time_bin_center)

if Station == 2:
    g_dim = 6

if Station == 3:
    g_dim = 7

ratio_r_mwx_map = np.full((ratio_len, r_len, 16, g_dim), 0, dtype = int)
ratio_time_mwx_map = np.full((ratio_len, time_len, 16, g_dim), 0, dtype = int)
r_time_mwx_map = np.full((r_len, time_len, g_dim), 0, dtype = int) 

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
print(unix_mwx)
print(np.degrees(lat_mwx))
print(np.degrees(lon_mwx))
print(r_mwx)

ara_geom = ROOT.AraGeomTool.Instance()
stationVector = ara_geom.getStationVector(Station)
st_long = ara_geom.getLongitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the longitude
st_lat = ara_geom.getGeometricLatitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the latitude
fGeoidC=6356752.3
fIceThicknessSP=2646.28
st_r = fGeoidC + fIceThicknessSP
print(st_long)
print(st_lat)
print(st_r)
st_long = np.radians(st_long)
st_lat = np.radians(st_lat)

ara_R2 = st_r**2
mwx_R2 = r_mwx**2
two_Rs = 2*r_mwx*st_r
tri_term = np.cos(st_lat) * np.cos(lat_mwx) * np.cos(st_long - lon_mwx) + np.sin(st_lat) * np.sin(lat_mwx)
radius = np.sqrt(ara_R2 + mwx_R2 - two_Rs * tri_term)
print(radius)

print(np.nanmax(radius))

print()
print(ara_R2)
print(mwx_R2[0])
print(two_Rs[0])
print(tri_term[0])
print(radius[0])
print()
print(np.degrees(st_lat))
print(np.degrees(lat_mwx[0]))
print(np.degrees(st_long))
print(np.degrees(lon_mwx[0]))
print()

ff_mwx = interp1d(unix_mwx, radius)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  #if r > 10001:

    #if d_run_tot[r] in bad_runs:
    #    #print('bad run:', d_list[r], d_run_tot[r])
    #    continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    if g_idx < 5:
        continue

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['clean_unix'][:]
    sub_ratio = np.nanmax(hf['sub_ratio'][:], axis = 0)    
    unix_idx = np.in1d(unix_time, unix_mwx)
    if np.count_nonzero(unix_idx) == 0:
        continue
    sub_ratio_cut = sub_ratio[:, unix_idx]

    unix_cut = unix_time[unix_idx]
    time_cut = (unix_cut - unix_cut[0]) / 60
    r_cut = ff_mwx(unix_cut)

    r_time_mwx_map[:,:,g_idx] += np.histogram2d(r_cut, time_cut, bins = (r_bins, time_bins))[0].astype(int)

    for a in range(16):
        ratio_r_mwx_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], r_cut, bins = (ratio_bins, r_bins))[0].astype(int)
        ratio_time_mwx_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], time_cut, bins = (ratio_bins, time_bins))[0].astype(int)
    del unix_idx, sub_ratio_cut, unix_cut, r_cut
    del hf, unix_time, sub_ratio

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_MWX_R_{d_type}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('time_bin_center', data=time_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_time_mwx_map', data=ratio_time_mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_r_mwx_map', data=ratio_r_mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('r_time_mwx_map', data=r_time_mwx_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






