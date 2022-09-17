##
# @file event_rate_calculator.py
#
# @section Created on 09/14/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to calculate event rate from 100 % (blind) data

import os
import numpy as np
import h5py
import click # 'pip3 install click' will make you very happy
import uproot # 'pip3 install uproot' will also make you very happy

@click.command()
@click.option('-d', '--data_path', type = str, help = 'ex) /misc/disk20/data/exp/ARA/2014/blinded/L1/ARA02/1026/run004434/event004434.root')
@click.option('-o', '--output_path', type = str, help = 'ex) /home/mkim/')
@click.option('-t', '--time_width', default = 60, type = int, help = 'ex) 60, 30, 1, etc')
@click.option('-p', '--use_pps', default = False, type = bool, help = 'ex) 0 or 1')
def main(data_path, output_path, time_width, use_pps):
    """! This is designed to calculate event rate from 100 % (blind) data
        Results of this calculation will only look reasonable if user use 100 % (blind) data

    @param data_path  string. data path
    @param output_path. string. output path
    @param time_width  integer. 
    @param use_pps  boolean.
    """
   
    ## load data by uproot
    file = uproot.open(data_path)
    evtTree = file['eventTree']

    ## identify station id and run number
    station_id = int(np.asarray(evtTree['event/RawAraStationEvent/RawAraGenericHeader/stationId'], dtype = int)[0])
    if station_id == 100: 
        station_id = 1 # ARA1....
    run_number = int(np.asarray(evtTree['run'], dtype = int)[0]) 
    print(f'Station id: {station_id}, Run number: {run_number}')

    ## load data that need to identify trigger type. copy from AraRoot
    trigger_info = np.asarray(evtTree['event/triggerInfo[4]'], dtype=int) # for software trigger
    time_stamp = np.asarray(evtTree['event/timeStamp'],dtype=int) # for calpulser trigger
    pulser_time = np.array([254, 245, 245, 400, 400], dtype = int) # calpulser length??
  
    ## get trigger type. 0: RF, 1: Calpulser, 2: Software  
    trig_type = np.full(len(time_stamp), 0, dtype = int)
    trig_type[np.abs(time_stamp - pulser_time[station_id - 1]) < 1e4] = 1
    trig_type[trigger_info[:, 2] == 1] = 2
    del pulser_time, trigger_info, time_stamp

    ## choose time type
    if use_pps:
        time = np.asarray(evtTree['event/ppsNumber'], dtype = int)
        pps_reset_idx = np.where(np.diff(time) < -55000)[0] # check whether pps was reset during the data taking
        if len(pps_reset_idx) > 0:
            pps_limit = 65536
            time[pps_reset_idx[0] + 1:] += pps_limit
    else:
        time = np.asarray(evtTree['event/unixTime'], dtype = int) 
    
    ## set time bins
    time_bins = np.arange(np.nanmin(time), np.nanmax(time) + 1, time_width, dtype = int)
    time_bins = time_bins.astype(float)
    time_bins -= 0.5 # set the boundary of binspace to between seconds. probably doesn't need it though...
    time_bins = np.append(time_bins, np.nanmax(time) + 0.5) # take into account last time bin which is smaller than time_width
    num_of_secs = np.diff(time_bins) # how many seconds in each bin space

    ## calculate event rate by mighty numpy histogram
    evt_rate = np.histogram(time, bins = time_bins)[0] / num_of_secs
    rf_evt_rate = np.histogram(time[trig_type == 0], bins = time_bins)[0] / num_of_secs
    cal_evt_rate = np.histogram(time[trig_type == 1], bins = time_bins)[0] / num_of_secs
    soft_evt_rate = np.histogram(time[trig_type == 2], bins = time_bins)[0] / num_of_secs
    del time, trig_type
 
    ## create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf_file_name = f'{output_path}Event_Rate_A{station_id}_R{run_number}.h5'
    hf = h5py.File(hf_file_name, 'w')
    hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
    hf.create_dataset('num_of_secs', data=num_of_secs, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
    hf.create_dataset('rf_evt_rate', data=rf_evt_rate, compression="gzip", compression_opts=9)
    hf.create_dataset('cal_evt_rate', data=cal_evt_rate, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_evt_rate', data=soft_evt_rate, compression="gzip", compression_opts=9)
    hf.close()

    print(f'Output is {hf_file_name}')
    print('Done!')

if __name__ == "__main__":

    main()















