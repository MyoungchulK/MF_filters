import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from scipy.interpolate import interp1d
import ROOT

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def get_dis(ara_Lat, ara_Lon, ara_R, Wgs84_lat, Wgs84_lon, Wgs84_r):
    
    ara_R2 = ara_R**2
    mwx_R2 = Wgs84_r**2
    two_Rs = 2 * ara_R * Wgs84_r
    
    tri_term = np.cos(ara_Lat) * np.cos(Wgs84_lat) * np.cos(ara_Lon - Wgs84_lon) + np.sin(ara_Lat) * np.sin(Wgs84_lat)
    radius = np.sqrt(ara_R2 + mwx_R2 - two_Rs * tri_term)
    
    return radius

def main(st = None, r_cut = 17000):

    # paths
    general_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/radiosonde_data/'
    h5_path = f'{general_path}h5/'
    r_path = f'{general_path}radius_tot/'
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    
    # h5 list
    mwx_list = glob(f'{h5_path}*')
    mwx_len = len(mwx_list)
    print('# of total h5 files:', mwx_len)

    # ara coord
    fGeoidC=6356752.3
    fIceThicknessSP=2646.28
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libRootFftwWrapper.so.3.0.1")
    ara_geom = ROOT.AraGeomTool.Instance()
    stationVector = ara_geom.getStationVector(st)
    ara_Lon = ara_geom.getLongitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the longitude
    ara_Lat = ara_geom.getGeometricLatitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) #get the latitude
    ara_Lat = np.radians(ara_Lat)
    ara_Lon = np.radians(ara_Lon)
    ara_R = fGeoidC + fIceThicknessSP
    print(f'A{st} coord. Lat: {np.degrees(ara_Lat)} deg, Lon: {np.degrees(ara_Lon)} deg, R: {ara_R} m')
    del fGeoidC, fIceThicknessSP, ara_geom, stationVector 

    # mwx coord
    pad = 18000
    unix_whole_flight = np.full((pad, mwx_len), np.nan, dtype = float)
    radius_whole_flight = np.copy(unix_whole_flight)

    for m in tqdm(range(mwx_len)):
        
        hf = h5py.File(mwx_list[m], 'r')
        SystemEvents = hf['SystemEvents']
        sys_unix = SystemEvents['UnixTime'][:]
        sys_evt_type = SystemEvents['EventType'][:]
        if mwx_list[m] == f'{h5_path}NZSP_20180310_211810.h5': # NZSP_20180310_211810.h5
        #if m == 122: # NZSP_20180310_211810.h5
            print(mwx_list[m])
            sys_unix = sys_unix[:-17]
            sys_evt_type = sys_evt_type[:-17]

        # PRE LAUNCH
        sys_evt_type = list(sys_evt_type.astype(str))
        launch_idx = sys_evt_type.index('BalloonReleased')
        sys_unix_launch = sys_unix[:launch_idx]
        pre_launch = np.arange(np.floor(np.nanmin(sys_unix_launch)), np.ceil(np.nanmax(sys_unix_launch)) + 1, 1, dtype = int)

        # r calculation
        GpsResults = hf['GpsResults']
        Wgs84X = GpsResults['Wgs84X'][:]
        Wgs84Y = GpsResults['Wgs84Y'][:]
        Wgs84Z = GpsResults['Wgs84Z'][:]
        gps_unix = GpsResults['UnixTime'][:]
        hf.close()
    
        Wgs84_lon, Wgs84_lat, Wgs84_r = cart2sph(Wgs84X, Wgs84Y, Wgs84Z)
        radius = get_dis(ara_Lat, ara_Lon, ara_R, Wgs84_lat, Wgs84_lon, Wgs84_r)

        dis_int = interp1d(np.append(gps_unix, gps_unix[-1]+1), np.append(radius, radius[-1]), fill_value = 'extrapolate')
        unix_pad = np.arange(np.floor(np.nanmin(sys_unix)), np.ceil(np.nanmax(sys_unix)) + 1, 1, dtype = int)
        radius_int = dis_int(unix_pad)
    
        pre_launch_idx = np.in1d(unix_pad, pre_launch)
        radius_int[pre_launch_idx] = 0

        unix_whole_flight[:len(unix_pad), m] = unix_pad
        radius_whole_flight[:len(radius_int), m] = radius_int
    
    bad_r_idx = radius_whole_flight < r_cut
    unix_temp = np.copy(unix_whole_flight)
    unix_temp[~bad_r_idx] = np.nan
    cw_unix_time = unix_temp[~np.isnan(unix_temp)]

    medi_name = f'{r_path}A{st}_mwx_R.h5'
    print(medi_name)
    hf2 = h5py.File(medi_name, 'w')
    hf2.create_dataset('unix_whole_flight', data=unix_whole_flight, compression="gzip", compression_opts=9)
    hf2.create_dataset('radius_whole_flight', data=radius_whole_flight, compression="gzip", compression_opts=9)
    hf2.create_dataset('cw_unix_time', data=cw_unix_time, compression="gzip", compression_opts=9)
    hf2.close()
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 3:
        Usage = """

    Usage = python3 %s <st ex)2 ro 3> <r cut ex)17000>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #argv
    ST = int(sys.argv[1])
    R_CUT = int(sys.argv[2])

    main(ST, R_CUT)















