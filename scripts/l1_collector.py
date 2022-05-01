import numpy as np
import os, sys
from tqdm import tqdm
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader
from tools.ara_run_manager import config_info_loader
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

def l1_loader(Station = None):

    print('Collecting l1 starts!')

    arr_len = 20000
    run_arr = np.full((arr_len), np.nan, dtype = float)
    run_type = np.copy(run_arr)
    evt_start_unix = np.copy(run_arr)
    evt_stop_unix = np.copy(run_arr)
    config_start_unix = np.copy(run_arr)
    config_stop_unix = np.copy(run_arr)

    print('pedestal!')
    batch_info = batch_info_loader(Station)
    ara_config = config_info_loader(verbose = False)
    yrs_arr = batch_info.years
    for yrs in tqdm(yrs_arr):
        if int(yrs) == 2013:
            continue
        ped_path = f'/data/exp/ARA/{int(yrs)}/calibration/pedestals/ARA0{Station}/'    
        start_run_num_yrs, start_unix_time_yrs, start_date_time_yrs, stop_run_num_yrs, stop_unix_time_yrs, stop_date_time_yrs = ara_config.get_ped_start_n_stop(ped_path)

        run_arr[start_run_num_yrs] = start_run_num_yrs
        run_type[start_run_num_yrs] = 0
        config_start_unix[start_run_num_yrs] = start_unix_time_yrs
        run_arr[stop_run_num_yrs] = stop_run_num_yrs
        run_type[stop_run_num_yrs] = 0
        config_stop_unix[stop_run_num_yrs] = stop_unix_time_yrs    
        del ped_path, start_run_num_yrs, start_unix_time_yrs, start_date_time_yrs, stop_run_num_yrs, stop_unix_time_yrs, stop_date_time_yrs
    del yrs_arr

    print('unblined event!')
    run_num, evt_path = batch_info.get_dat_list(analyze_blind_dat = False)[:2]
    run_num = run_num.astype(int)
    run_arr[run_num] = run_num
    run_type[run_num] = 1

    for run in tqdm(range(len(run_num))):
        slash_idx = evt_path[run].rfind('/')
        run_path = evt_path[run][:slash_idx] + '/'
        unix_time, date_time = ara_config.get_run_start_n_stop(run_path)
        if ~np.isnan(unix_time[0]):
            config_start_unix[run_num[run]] = unix_time[0]
        if ~np.isnan(unix_time[1]):
            config_stop_unix[run_num[run]] = unix_time[1]
        del slash_idx, run_path, unix_time, date_time
    del run_num, evt_path

    print('blinded event!')
    run_num, evt_path = batch_info.get_dat_list(analyze_blind_dat = True)[:2]
    run_num = run_num.astype(int)
    run_arr[run_num] = run_num
    run_type[run_num] = 1

    for run in tqdm(range(len(run_num))):
        slash_idx = evt_path[run].rfind('/')
        run_path = evt_path[run][:slash_idx] + '/'
        unix_time, date_time = ara_config.get_run_start_n_stop(run_path)
        if ~np.isnan(unix_time[0]):
            config_start_unix[run_num[run]] = unix_time[0]
        if ~np.isnan(unix_time[1]):
            config_stop_unix[run_num[run]] = unix_time[1]
        del slash_idx, run_path, unix_time, date_time

    for run in tqdm(range(len(run_num))):

        try:
            run_time_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/run_time_full/run_time_full_A{Station}_R{run_num[run]}.h5'
            hf = h5py.File(run_time_path, 'r')   
            evt_unix = hf['evt_unix'][:] 
            evt_start_unix[run_num[run]] = evt_unix[0]
            evt_stop_unix[run_num[run]] = evt_unix[1]
            del run_time_path, hf, evt_unix
        except FileNotFoundError:
            print(f'A{Station} R{run_num[run]} is missing!')
    del batch_info, run_num, evt_path, ara_config

    config_run_time = config_stop_unix - config_start_unix
    evt_run_time = evt_stop_unix - evt_start_unix

    knwon_issue = known_issue_loader(Station)
    bad_runs = knwon_issue.get_knwon_bad_run()
    del knwon_issue

    run_arr_cut = np.copy(run_arr)
    run_type_cut = np.copy(run_type)
    evt_start_unix_cut = np.copy(evt_start_unix)
    evt_stop_unix_cut = np.copy(evt_stop_unix)
    config_start_unix_cut = np.copy(config_start_unix)
    config_stop_unix_cut = np.copy(config_stop_unix)
    config_run_time_cut = np.copy(config_run_time)
    evt_run_time_cut = np.copy(evt_run_time)

    for c in tqdm(range(arr_len)):
      if c in bad_runs:
        run_arr_cut[c] = np.nan
        run_type_cut[c] = np.nan
        evt_start_unix_cut[c] = np.nan
        evt_stop_unix_cut[c] = np.nan
        config_start_unix_cut[c] = np.nan
        config_stop_unix_cut[c] = np.nan
        config_run_time_cut[c] = np.nan
        evt_run_time_cut[c] = np.nan
    del arr_len

    print('L1 collecting is done!')

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}L1_Goal_A{Station}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('run_type', data=run_type, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_start_unix', data=evt_start_unix, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_stop_unix', data=evt_stop_unix, compression="gzip", compression_opts=9)
    hf.create_dataset('config_start_unix', data=config_start_unix, compression="gzip", compression_opts=9)
    hf.create_dataset('config_stop_unix', data=config_stop_unix, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_run_time', data=evt_run_time, compression="gzip", compression_opts=9) 
    hf.create_dataset('config_run_time', data=config_run_time, compression="gzip", compression_opts=9) 
    hf.create_dataset('run_arr_cut', data=run_arr_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('run_type_cut', data=run_type_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_start_unix_cut', data=evt_start_unix_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_stop_unix_cut', data=evt_stop_unix_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('config_start_unix_cut', data=config_start_unix_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('config_stop_unix_cut', data=config_stop_unix_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_run_time_cut', data=evt_run_time_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('config_run_time_cut', data=config_run_time_cut, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
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
    
    l1_loader(Station = station)
 
