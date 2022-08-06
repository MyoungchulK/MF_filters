##
# @file balloon_unixtime_calculator.py
#
# @section Created on 08/06/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to calculate distance between weather balloon and each station and corresponding unix time

import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from scipy.interpolate import interp1d
import ROOT

def cart_to_sph(x, y, z):
    """! cartesian coordinates to spherical coordinates

    @param x  float. meter
    @param y  float. meter
    @param z  float. meter
    @return az  float. radian
    @return el  float. radian
    @return r  float. meter
    """

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)

    return az, el, r

def get_distance(lat1, lon1, r1, lat2, lon2, r2):
    """! get distance from spherical coordinates

    @param lat1  float. radian
    @param lon1  float. radian
    @param r1  float. meter
    @param lat2  float. radian
    @param lon2  float. radian
    @param r2  float. meter
    @return distance  float. meter 
    """    

    r1_sq = r1**2
    r2_sq = r2**2
    r12 = 2 * r1 * r2
    tri_term = np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2) + np.sin(lat1) * np.sin(lat2)
    distance = np.sqrt(r1_sq + r2_sq - r12 * tri_term)
    
    return distance

def main(h5_path, output_path, st, distance_cut = 17000):
    """! main function for calculating distance between weather balloon and each station and corresponding unix time

    @param h5_path  string
    @param output_path  string
    @param st  integer. station id
    @param distance_cut  float
    """

    ## ARA station coordinate
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
   
    ## load longitude and latitude from AraRoot. Imported from Biran's repo 
    ara_geom = ROOT.AraGeomTool.Instance()
    stationVector = ara_geom.getStationVector(st)
    ara_Lon = ara_geom.getLongitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) # get the longitude
    ara_Lat = ara_geom.getGeometricLatitudeFromArrayCoords(stationVector[1], stationVector[0], 2011) # get the 'Geometric' latitude. We are going to compare with weather balloon 'XYZ' coordinates
    ara_Lat = np.radians(ara_Lat)
    ara_Lon = np.radians(ara_Lon) 

    ## These value are copied from AraRoot. In the future, we need make a function in AraRoot to return these values
    fGeoidC=6356752.3
    fIceThicknessSP=2646.28 
    ara_R = fGeoidC + fIceThicknessSP
    print(f'A{st} coord. Lat: {np.degrees(ara_Lat)} deg, Lon: {np.degrees(ara_Lon)} deg, R: {ara_R} m') 

    ## make h5 file list in h5_list
    h5_list = glob(f'{h5_path}*')
    h5_len = len(h5_list)
    print('# of total h5 files:', h5_len)

    ## numpy array pad for saving disance and unix time of evil balloon
    pad = 18000
    balloon_unix_time = np.full((pad, h5_len), np.nan, dtype = float)
    balloon_distance = np.copy(balloon_unix_time)

    ## loop over all h5 files
    for h in tqdm(range(h5_len)):
      #if h < 10: # for debug        

        hf = h5py.File(h5_list[h], 'r')

        ## tag pre launch period as a bad unix time: time between 'viselar on' and 'balloon released'
        ## Since in this time period, all the GPS informations are showing error value and distance to all stations are close, I dont think this time period is safe to use for analysis
        SystemEvents = hf['SystemEvents']
        sys_unix = SystemEvents['UnixTime'][:]
        sys_evt_type = SystemEvents['EventType'][:]
        if h5_list[h].find('NZSP_20180310_211810') != -1: # corrupted unxitime. evil...
            print('NZSP_20180310_211810, corrupted unxitime')
            sys_unix = sys_unix[:-17]
            sys_evt_type = sys_evt_type[:-17]

        ## find the unix time when 'balloon released'
        sys_evt_type = list(sys_evt_type.astype(str))
        try:
            str_idx = sys_evt_type.index('BalloonReleased')
            sys_unix_launch = sys_unix[:str_idx]
        except ValueError:
            print(f'There is no BalloonReleased tag in {h5_list[h]}. Condamn all this flight time')
            sys_unix_launch = np.copy(sys_unix)
        pre_launch = np.arange(np.floor(np.nanmin(sys_unix_launch)), np.ceil(np.nanmax(sys_unix_launch)) + 1, 1, dtype = int) # final pre launch unix time 

        ## load balloon XYZ and unix time
        GpsResults = hf['GpsResults']
        Wgs84X = GpsResults['Wgs84X'][:]
        Wgs84Y = GpsResults['Wgs84Y'][:]
        Wgs84Z = GpsResults['Wgs84Z'][:]
        gps_unix = GpsResults['UnixTime'][:]
        hf.close()
   
        ## distance calculation 
        Wgs84_lon, Wgs84_lat, Wgs84_r = cart_to_sph(Wgs84X, Wgs84Y, Wgs84Z) # converted to 1) longitude, 2) 'Geometric' latitude, 3) Radius
        distance = get_distance(ara_Lat, ara_Lon, ara_R, Wgs84_lat, Wgs84_lon, Wgs84_r)

        ## fill the time gap by interpolation
        dis_int = interp1d(np.append(gps_unix, gps_unix[-1]+1), np.append(distance, distance[-1]), fill_value = 'extrapolate') # duplicate fianl value to array for preventing wrong extrapolation
        tot_unix_time = np.arange(np.floor(np.nanmin(sys_unix)), np.ceil(np.nanmax(sys_unix)) + 1, 1, dtype = int) # total balloon operation time including when balloon is in the ground
        tot_distance = dis_int(tot_unix_time)
   
        ## replace distance to 0 at the pre launch period. so that no matter what distance_cut user use, it will always tagged as a bad period for analysis 
        pre_launch_idx = np.in1d(tot_unix_time, pre_launch)
        tot_distance[pre_launch_idx] = 0

        ## save into numpy array 
        balloon_unix_time[:len(tot_unix_time), h] = tot_unix_time
        balloon_distance[:len(tot_distance), h] = tot_distance
    
    ## calculating bad unix time for analysis
    bad_dis_idx = balloon_distance < distance_cut
    bad_unix_time = np.copy(balloon_unix_time)
    bad_unix_time[~bad_dis_idx] = np.nan
    bad_unix_time = bad_unix_time[~np.isnan(bad_unix_time)]
    bad_unix_time = bad_unix_time.astype(int)
    bad_unix_time = np.unique(bad_unix_time).astype(int)

    ## create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(f'{output_path}A{st}_balloon_distance.h5', 'w')
    hf.create_dataset('balloon_unix_time', data=balloon_unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('balloon_distance', data=balloon_distance, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_unix_time', data=bad_unix_time, compression="gzip", compression_opts=9)
    hf.close()
    print('Done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 4 and len (sys.argv) != 5:
        Usage = """

    Usage = python3 %s <h5_path ex)/data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/h5/> <output_path ex)/data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/> <st ex)2> <distance = 17000>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #argv
    H5_path = str(sys.argv[1])
    Output_path = str(sys.argv[2])
    ST = int(sys.argv[3])
    if len (sys.argv) == 5:
        DIS_CUT = int(sys.argv[4])
    else:
        DIS_CUT = 17000 
    print("H5 Path: {}, Output Path: {}, Station ID: {}, Distance Cut: {}".format(H5_path, Output_path, ST, DIS_CUT))

    main(h5_path = H5_path, output_path = Output_path, st = ST, distance_cut = DIS_CUT)















