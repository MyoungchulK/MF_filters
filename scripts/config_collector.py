import numpy as np
import os, sys
from tqdm import tqdm
from datetime import datetime
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager_temp import batch_info_loader
from tools.ara_run_manager_temp import config_info_loader
from tools.ara_run_manager_temp import run_info_loader
from tools.ara_utility import size_checker

def unix_to_date(unix):

    date = datetime.fromtimestamp(unix)
    date = date.strftime('%Y%m%d%H%M%S')
    date = int(date)    

    return date

def config_loader(Station = None):

    print('Collecting config starts!')

    year_edge = (2014,2020)

    analyze_blind_dat = False
    batch_info = batch_info_loader(Station, year = year_edge)
    run_num, evt_dat_path = batch_info.get_dat_list(analyze_blind_dat = analyze_blind_dat)[:2]
    num_runs = 20000
    ara_config = config_info_loader(verbose = True)

    yrs_range = np.arange(year_edge[0], year_edge[-1], dtype = int)
    start_run_num = np.array([], dtype = int)
    start_unix_time = np.array([], dtype = float)
    start_date_time = np.array([], dtype = float)
    stop_run_num = np.array([], dtype = int)
    stop_unix_time = np.array([], dtype = float)
    stop_date_time = np.array([], dtype = float)
    for y in tqdm(range(len(yrs_range))):
        ped_path = f'/data/exp/ARA/{yrs_range[y]}/calibration/pedestals/ARA0{Station}/'
        start_run_num_yrs, start_unix_time_yrs, start_date_time_yrs, stop_run_num_yrs, stop_unix_time_yrs, stop_date_time_yrs = ara_config.get_ped_start_n_stop(ped_path)
        start_run_num = np.append(start_run_num, start_run_num_yrs)
        start_unix_time = np.append(start_unix_time, start_unix_time_yrs)
        start_date_time = np.append(start_date_time, start_date_time_yrs)
        stop_run_num = np.append(stop_run_num, stop_run_num_yrs)
        stop_unix_time = np.append(stop_unix_time, stop_unix_time_yrs)
        stop_date_time = np.append(stop_date_time, stop_date_time_yrs)

    run_arr = np.arange(num_runs, dtype = int)
    run_type = np.full((num_runs), np.nan, dtype = float)
    unix_time = np.full((2, num_runs), np.nan, dtype = float)
    date_time = np.copy(unix_time)
    evt_unix_time = np.copy(unix_time)
    evt_date_time = np.copy(unix_time)

    for run in tqdm(range(num_runs)):
        
        if run in run_num:
            run_type[run] = 1
            run_idx = np.where(run_num == run)[0][0]
            slash_idx = evt_dat_path[run_idx].rfind('/')
            run_path_dir = evt_dat_path[run_idx][:slash_idx] + '/'
            unix_time[:, run], date_time[:, run] = ara_config.get_run_start_n_stop(run_path_dir)
            del run_path_dir, slash_idx       
     
            run_info = run_info_loader(Station, run_num[run_idx], analyze_blind_dat = True)
            qual_path = run_info.get_result_path(file_type = 'qual_cut')
            qual = h5py.File(qual_path, 'r')
            qual_unix_time = qual['unix_time'][:]
            del run_idx, run_info, qual_path, qual
            
            evt_unix_start = qual_unix_time[0]
            evt_unix_stop = qual_unix_time[-1]
            evt_date_start = unix_to_date(evt_unix_start) 
            evt_date_stop = unix_to_date(evt_unix_stop) 
            evt_unix_time[0, run] = evt_unix_start
            evt_unix_time[-1, run] = evt_unix_stop
            evt_date_time[0, run] = evt_date_start
            evt_date_time[-1, run] = evt_date_stop
            del qual_unix_time, evt_unix_start, evt_unix_stop, evt_date_start, evt_date_stop

        if run in start_run_num:
            run_type[run] = 0
            run_idx = np.where(start_run_num == run)[0][0]
            unix_time[0, run] = start_unix_time[run_idx]
            date_time[0, run] = start_date_time[run_idx]

        if run in stop_run_num:
            run_type[run] = 0
            run_idx = np.where(stop_run_num == run)[0][0]
            unix_time[1, run] = stop_unix_time[run_idx]
            date_time[1, run] = stop_date_time[run_idx]

    print('Congfig collecting is done!')

    # create output dir
    Key = 'config'
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}{blind_type}/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}{Key}{blind_type}_A{Station}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('run_type', data=run_type, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('date_time', data=date_time, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_unix_time', data=evt_unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_date_time', data=evt_date_time, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)

if __name__ == "__main__":

    if len (sys.argv) != 2:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    
    config_loader(Station = station)
 
