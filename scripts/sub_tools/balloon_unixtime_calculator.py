##
# @file balloon_unixtime_calculator.py
#
# @section Created on 08/06/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to calculate distance between weather balloon and each station and corresponding unix time

import os, sys
import numpy as np
import h5py
import click # 'pip3 install click' will make you very happy
from tqdm import tqdm
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import medfilt
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

@click.command()
@click.option('-h', '--h5_path', type = str, help = 'ex) /data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/h5/')
@click.option('-o', '--output_path', type = str, help = 'ex) /data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/')
@click.option('-s', '--st', type = int, help = 'ex) 2')
@click.option('-d', '--distance_cut', default = 17000, type = float, help = 'ex) 17000')
def main(h5_path, output_path, st, distance_cut):
    """! main function for calculating distance between weather balloon and each station and corresponding unix time
        cuttently this code is including lots of hard coding. Please read the code before launch

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
    ara_R = ara_geom.getGeometricRadius()
    print(f'A{st} coord. Lat: {np.degrees(ara_Lat)} deg, Lon: {np.degrees(ara_Lon)} deg, R: {ara_R} m') 

    ## make h5 file list in h5_list
    h5_list = glob(f'{h5_path}*.h5')
    h5_len = len(h5_list)
    print('# of total h5 files:', h5_len)

    ## numpy array pad for saving disance and unix time of evil balloon
    pad = 25000
    balloon_unix_time = np.full((pad, h5_len), np.nan, dtype = float)
    balloon_distance = np.copy(balloon_unix_time)
    balloon_smooth_distance = np.copy(balloon_unix_time)

    ## h5 file tree name
    log_name = ['SystemEvents', 'metadata']
    log_label = [['UnixTime', 'EventType'], ['UnixTime', 'TimeLabel']]
    launch_name = ['BalloonReleased', 'RsActualLaunchTime']
    gps_name = ['GpsResults', 'GPSDCC_RESULT000']
    gps_label = [['Wgs84X', 'Wgs84Y', 'Wgs84Z', 'UnixTime'], ['dSondeX [m]', 'dSondeY [m]', 'dSondeZ [m]', 'time [s]']]

    ## loop over all h5 files
    for h in tqdm(range(h5_len)):
      #if h == 4162: # for debug        
        #print(h5_list[h]) # for debug

        hf = h5py.File(h5_list[h], 'r')

        ## check whether it is from pre 2016 or post 2016
        nzsp_flag = h5_list[h].find('NZSP')
        southpole_flag = h5_list[h].find('SOUTHPOLE')
        if nzsp_flag != -1:
            t_idx = 0
        elif southpole_flag != -1:
            t_idx = 1
        else:
            print(f'Something wrong in {h5_list[h]} !!')
            sys.exit(1)

        ## tag pre launch period as a bad unix time: time between 'viselar on' and 'balloon released'
        ## Since in this time period, all the GPS informations are showing error value and distance to all stations are close, I dont think this time period is safe to use for analysis
        SystemEvents = hf[log_name[t_idx]]
        sys_unix = SystemEvents[log_label[t_idx][0]][:]
        sys_evt_type = SystemEvents[log_label[t_idx][1]][:]
        if h5_list[h].find('NZSP_20180310_211810') != -1: # corrupted unxitime. evil...
            print('NZSP_20180310_211810, corrupted unxitime')
            sys_unix = sys_unix[:-17]
            sys_evt_type = sys_evt_type[:-17]

        ## find the unix time when 'balloon released'
        sys_evt_type = list(sys_evt_type.astype(str))
        try:
            str_idx = sys_evt_type.index(launch_name[t_idx])+1
            sys_unix_launch = sys_unix[:str_idx]
        except ValueError:
            print(f'There is no {launch_name[t_idx]} tag in {h5_list[h]}. Condamn all this flight time')
            sys_unix_launch = np.copy(sys_unix)
        if h5_list[h].find('SOUTHPOLE_20110331_211530') != -1 or h5_list[h].find('SOUTHPOLE_20121221_091513') != -1 or h5_list[h].find('SOUTHPOLE_20120118_093154') != -1 or h5_list[h].find('SOUTHPOLE_20121106_092614') != -1 or h5_list[h].find('SOUTHPOLE_20121107_084439') != -1 or h5_list[h].find('SOUTHPOLE_20120210_221021') != -1  or h5_list[h].find('SOUTHPOLE_20130123_212047') != -1 or h5_list[h].find('SOUTHPOLE_20131227_091019') != -1  or h5_list[h].find('SOUTHPOLE_20130220_094246') != -1 or h5_list[h].find('SOUTHPOLE_20131114_092949') != -1  or h5_list[h].find('SOUTHPOLE_20130506_221741') != -1 or h5_list[h].find('SOUTHPOLE_20141127_212335') != -1 or h5_list[h].find('SOUTHPOLE_20141128_222220') != -1 or h5_list[h].find('SOUTHPOLE_20141113_213258') != -1 or h5_list[h].find('SOUTHPOLE_20150223_095428') != -1 or h5_list[h].find('SOUTHPOLE_20150501_222129') != -1: # bad flights...
            print(f'Wrong time label in {h5_list[h]} !!')
            pre_launch = np.arange(np.floor(sys_unix_launch[2]), np.ceil(sys_unix_launch[3]) + 1, 1, dtype = int)
        else:
            pre_launch = np.arange(np.floor(np.nanmin(sys_unix_launch)), np.ceil(np.nanmax(sys_unix_launch)) + 1, 1, dtype = int) # final pre launch unix time 

        ## load balloon XYZ and unix time
        GpsResults = hf[gps_name[t_idx]]
        Wgs84X = GpsResults[gps_label[t_idx][0]][:]
        Wgs84Y = GpsResults[gps_label[t_idx][1]][:]
        Wgs84Z = GpsResults[gps_label[t_idx][2]][:]
        gps_unix = GpsResults[gps_label[t_idx][3]][:]
        if t_idx == 1:
            reset_idx = sys_evt_type.index('RsTimeResetBase')
            gps_unix += sys_unix[reset_idx]
        hf.close()

        ## distance calculation 
        Wgs84_lon, Wgs84_lat, Wgs84_r = cart_to_sph(Wgs84X, Wgs84Y, Wgs84Z) # converted to 1) longitude, 2) 'Geometric' latitude, 3) Radius
        distance = get_distance(ara_Lat, ara_Lon, ara_R, Wgs84_lat, Wgs84_lon, Wgs84_r)

        ## remove nan
        dis_nan = np.isnan(distance)
        gps_unix = gps_unix[~dis_nan]
        distance = distance[~dis_nan]
        if h5_list[h].find('SOUTHPOLE_20111107_212531') != -1:
            print(f'There is too much error in {h5_list[h]}. Surgical removing')
            distance[10:350] = np.nan
            dis_nan = np.isnan(distance)
            gps_unix = gps_unix[~dis_nan]
            distance = distance[~dis_nan]
        if h5_list[h].find('SOUTHPOLE_20130424_220017') != -1:
            print(f'There is too much error in {h5_list[h]}. Surgical removing')
            distance[9:] = np.nan
            dis_nan = np.isnan(distance)
            gps_unix = gps_unix[~dis_nan]
            distance = distance[~dis_nan]

        if h5_list[h].find('SOUTHPOLE_20110803_215627') != -1: # bad flights...
            print(f'There is no distance info. in {h5_list[h]}. Condamn all this flight time')
            temp_flight = np.arange(1000, dtype = int) + int(reset_idx)
            balloon_unix_time[:len(temp_flight), h] = temp_flight
            balloon_distance[:len(temp_flight), h] = 0
            balloon_smooth_distance[:len(temp_flight), h] = 0
            continue

        if len(distance) == 0:
            print(f'There is no distance info. in {h5_list[h]}. Condamn all this flight time')
            balloon_unix_time[:len(pre_launch), h] = pre_launch
            balloon_distance[:len(pre_launch), h] = 0
            balloon_smooth_distance[:len(pre_launch), h] = 0
            continue

        ## fill the time gap by interpolation
        dis_int = interp1d(np.append(gps_unix, gps_unix[-1]+1), np.append(distance, distance[-1]), fill_value = 'extrapolate') # duplicate fianl value to array for preventing wrong extrapolation
        if h5_list[h].find('SOUTHPOLE_20110331_211530') != -1 or h5_list[h].find('SOUTHPOLE_20121221_091513') != -1 or h5_list[h].find('SOUTHPOLE_20120118_093154') != -1 or h5_list[h].find('SOUTHPOLE_20121106_092614') != -1  or h5_list[h].find('SOUTHPOLE_20121107_084439') != -1 or h5_list[h].find('SOUTHPOLE_20120210_221021') != -1  or h5_list[h].find('SOUTHPOLE_20130123_212047') != -1 or h5_list[h].find('SOUTHPOLE_20131227_091019') != -1 or h5_list[h].find('SOUTHPOLE_20130220_094246') != -1 or h5_list[h].find('SOUTHPOLE_20131114_092949') != -1  or h5_list[h].find('SOUTHPOLE_20130506_221741') != -1 or h5_list[h].find('SOUTHPOLE_20141127_212335') != -1 or h5_list[h].find('SOUTHPOLE_20141128_222220') != -1 or h5_list[h].find('SOUTHPOLE_20141113_213258') != -1 or h5_list[h].find('SOUTHPOLE_20150223_095428') != -1 or h5_list[h].find('SOUTHPOLE_20150501_222129') != -1: # bad flights...
            max_unix = np.ceil(gps_unix[-1])
        else:
            max_unix = np.ceil(np.nanmax(sys_unix))
        tot_unix_time = np.arange(np.floor(np.nanmin(sys_unix)), max_unix + 1, 1, dtype = int) # total balloon operation time including when balloon is in the ground
        tot_distance = dis_int(tot_unix_time)
        tot_smooth_distance = medfilt(tot_distance, kernel_size = 39) # use rolling median to remove huge distance (wgs84xyz) value probably caused by error         
 
        ## replace distance to 0 at the pre launch period. so that no matter what distance_cut user use, it will always tagged as a bad period for analysis 
        pre_launch_idx = np.in1d(tot_unix_time, pre_launch)
        tot_distance[pre_launch_idx] = 0
        tot_smooth_distance[pre_launch_idx] = 0

        ## save into numpy array 
        balloon_unix_time[:len(tot_unix_time), h] = tot_unix_time
        balloon_distance[:len(tot_distance), h] = tot_distance
        balloon_smooth_distance[:len(tot_smooth_distance), h] = tot_smooth_distance

    ## calculating bad unix time for analysis
    bad_dis_idx = balloon_smooth_distance < distance_cut
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
    hf.create_dataset('balloon_smooth_distance', data=balloon_smooth_distance, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_unix_time', data=bad_unix_time, compression="gzip", compression_opts=9)
    hf.close()
    print('Done!')

if __name__ == "__main__":

    main()















