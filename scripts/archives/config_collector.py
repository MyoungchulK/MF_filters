import numpy as np
import os, sys
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

def context_finder(config_file, key, end_key, empty):

    val_i = config_file.find(key)
    if val_i != -1:
        val_i += len(key)
        val_f = config_file.find(end_key,val_i)
        if key == 'enableL1Trigger#I20=' or key == 'triggerDelays#I16=':
            val = np.asarray(config_file[val_i:val_f].split(",")).astype(np.int)
        else:
            val = int(config_file[val_i:val_f])
        del val_f
    else:
        val = empty
    del val_i

    return val

def run_time_start_info(run_start_list, run_key, empty_run_key, time_key, empty_time_key, run_end_key):

    # run info
    with open(run_start_list,'r') as ri_file:
        run_start = ri_file.read()

        run_start_num = context_finder(run_start, run_key, time_key, empty_run_key)
        if not np.isfinite(run_start_num):
            print('run_start_num is Nan!! Use file name for identifying run number!')
            print(run_start_num)
            run_start_num = int(run_start_list[-10:-4])
            print(run_start_num)
        else:
            pass
        unix_start = context_finder(run_start, time_key, run_end_key, empty_time_key)
        if not np.isfinite(unix_start):
            print('No runStart!')
            date_start = np.nan
        else:
            date_start = datetime.fromtimestamp(unix_start)
            date_start = date_start.strftime('%Y%m%d%H%M%S')
            date_start = int(date_start)
    
    return run_start_num, unix_start, date_start

def run_time_stop_info(run_stop_list, run_key, empty_run_key, time_key, empty_time_key, run_end_key):

    with open(run_stop_list,'r') as rf_file:
        run_stop = rf_file.read()
    
        run_stop_num = context_finder(run_stop, run_key, time_key, empty_run_key)
        if not np.isfinite(run_stop_num):
            print('run_stop_num is Nan!! Use file name for identifying run number!')
            print(run_stop_num)
            run_stop_num = int(run_stop_list[-10:-4])
            print(run_stop_num)
        else:
            pass
        unix_stop = context_finder(run_stop, time_key, run_end_key, empty_time_key)
        if not np.isfinite(unix_stop):
            print('No runStop!')
            date_stop = np.nan
        else:
            date_stop = datetime.fromtimestamp(unix_stop)
            date_stop = date_stop.strftime('%Y%m%d%H%M%S')
            date_stop = int(date_stop)
    
    return run_stop_num, unix_stop, date_stop

def config_loader(CPath = curr_path, Station = None, Output = None):

    print('Configuration starts!')

    print('Collectiong file path!')
    config_list = []
    run_start_list = []
    run_stop_list = []
    Year = np.arange(2013,2019)
    for y in tqdm(range(len(Year))):
        if Year[y] == 2013:
            #g_path = f'/data/exp/ARA/{Year[y]}/filtered/unzippedTGZFiles/ARA0{Station}/run_[0-9]*/logs/'
            g_path = f'/data/exp/ARA/{Year[y]}/raw/ARA0{Station}-SPS-ARA/run_[0-9]*/logs/'
            config_list += glob(f'{g_path}configFile.run[0-9]*.dat')
            run_start_list += glob(f'{g_path}runStart.run[0-9]*.dat')
            run_stop_list += glob(f'{g_path}runStop.run[0-9]*.dat')
        else:
            if Station == 3 and Year[y] == 2017:
                pass
            else:
                g_path = f'/data/exp/ARA/{Year[y]}/unblinded/L1/ARA0{Station}/*/run*/'
                config_list += glob(f'{g_path}configFile.run[0-9]*.dat')
                run_start_list += glob(f'{g_path}runStart.run[0-9]*.dat')
                run_stop_list += glob(f'{g_path}runStop.run[0-9]*.dat')
    run_len = len(config_list)
    runstart_len = len(run_start_list)
    runstop_len = len(run_stop_list)
    print('Total config:',run_len)
    print('Total runstart:',runstart_len)
    print('Total runstop:',runstop_len)

    print('Collecting information')
    run_start_num = []
    run_stop_num = []
    start_time_unix = []
    stop_time_unix = []
    start_time_date = []
    stop_time_date = []
    masked_ant = []
    rf_block_num = []
    soft_block_num = []
    trig_win_num = []
    delay_enable = []
    delay_num = []

    # keys
    run_key = 'Run:'
    empty_run_key = np.nan

    time_key = 'Time:'
    empty_time_key = np.nan    

    run_end_key = 'Message'

    end_key = ';'
    mask_key = 'enableL1Trigger#I20='
    empty_l1_mask = np.full((20),np.nan)

    rf_key = 'numRF0TriggerBlocks#1='
    empty_rf_block = np.nan

    soft_key = 'numSoftTriggerBlocks#1='
    empty_soft_block = np.nan

    trig_key = 'triggerWindowSize#1='
    empty_trig_win = np.nan

    delay_enable_key = 'enableTriggerDelays#I1='
    empty_delay_enable = np.nan

    delay_key = 'triggerDelays#I16='
    empty_delay = np.full((16),np.nan) 
    
    for runs in tqdm(range(runstart_len)):
        
        r_start_num, u_start, d_start = run_time_start_info(run_start_list[runs], run_key, empty_run_key, time_key, empty_time_key, run_end_key)
        run_start_num.append(r_start_num)
        start_time_unix.append(u_start)
        start_time_date.append(d_start)
        del r_start_num, u_start, d_start
  
    for runs in tqdm(range(runstop_len)):

        r_stop_num, u_stop, d_stop = run_time_stop_info(run_stop_list[runs], run_key, empty_run_key, time_key, empty_time_key, run_end_key)
        run_stop_num.append(r_stop_num)
        stop_time_unix.append(u_stop)
        stop_time_date.append(d_stop)
        del r_stop_num, u_stop, d_stop
 
    config_run = []

    for runs in tqdm(range(run_len)):

        config_run.append(int(config_list[runs][-10:-4]))

        with open(config_list[runs],'r') as config_file_txt:
            config_file = config_file_txt.read()

            masked_ant.append(context_finder(config_file, mask_key, end_key, empty_l1_mask))
            rf_block_num.append(context_finder(config_file, rf_key, end_key, empty_rf_block))
            soft_block_num.append(context_finder(config_file, soft_key, end_key, empty_soft_block))
            trig_win_num.append(context_finder(config_file, trig_key, end_key, empty_trig_win))
            delay_enable.append(context_finder(config_file, delay_enable_key, end_key, empty_delay_enable))
            delay_num.append(context_finder(config_file, delay_key, end_key, empty_delay))

    config_run = np.asarray(config_run)
    masked_ant = np.transpose(np.asarray(masked_ant),(1,0))
    rf_block_num = np.asarray(rf_block_num)
    soft_block_num = np.asarray(soft_block_num)
    trig_win_factor = np.array([10])
    trig_win_num = np.asarray(trig_win_num)
    delay_enable = np.asarray(delay_enable)
    delay_factor = np.array([10])
    delay_num = np.transpose(np.asarray(delay_num),(1,0))

    run_start_num = np.asarray(run_start_num)
    run_stop_num = np.asarray(run_stop_num)
    start_time_unix = np.asarray(start_time_unix)
    stop_time_unix = np.asarray(stop_time_unix)
    start_time_date = np.asarray(start_time_date)
    stop_time_date = np.asarray(stop_time_date)


    run_num = np.copy(config_run)
    unix_time = np.full((2,len(run_num)),np.nan)
    date_time = np.full((2,len(run_num)),np.nan)
    for r in tqdm(range(len(run_num))):
        start_idx = np.where(run_start_num == run_num[r])[0]
        if len(start_idx) == 1:
            unix_time[0,r] = start_time_unix[start_idx]
            date_time[0,r] = start_time_date[start_idx]
        else:
            pass
        stop_idx = np.where(run_stop_num == run_num[r])[0]
        if len(stop_idx) == 1:
            unix_time[1,r] = stop_time_unix[stop_idx]
            unix_time[1,r] = stop_time_date[stop_idx]
        else:
            pass

    print('Sorting starts!')
    run_index = np.argsort(run_num)
    
    run_num = run_num[run_index]
    unix_time = unix_time[:, run_index]
    date_time = date_time[:, run_index]
    masked_ant = masked_ant[:, run_index]
    rf_block_num = rf_block_num[run_index]
    soft_block_num = soft_block_num[run_index]
    trig_win_num = trig_win_num[run_index]
    delay_enable = delay_enable[run_index]
    delay_num = delay_num[:, run_index]
    del run_index
    print(run_num)
    print(unix_time)
    print(date_time)
    print(masked_ant)
    print(rf_block_num)
    print(soft_block_num)
    print(trig_win_num)
    print(delay_enable)
    print(delay_num)
    print('total run:',len(run_num))

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Config_A{Station}.h5'
    hf = h5py.File(h5_file_name, 'w')
    hf.create_dataset('run_num', data=run_num, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('date_time', data=date_time, compression="gzip", compression_opts=9)
    hf.create_dataset('masked_ant', data=masked_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('rf_block_num', data=rf_block_num, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_block_num', data=soft_block_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_win_num', data=trig_win_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_win_factor', data=trig_win_factor, compression="gzip", compression_opts=9)
    hf.create_dataset('delay_enable', data=delay_enable, compression="gzip", compression_opts=9)
    hf.create_dataset('delay_num', data=delay_num, compression="gzip", compression_opts=9)
    hf.create_dataset('delay_factor', data=delay_factor, compression="gzip", compression_opts=9)
    hf.close() 

    print(f'output is {Output}{h5_file_name}')
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 3:
        Usage = """
    This is designed to collect data from configFile.run[xxxxxx].dat, runStart.run[xxxxxx].dat, and runStop.run[xxxxxx].dat
    that required for setting configuration for each run.
    Example data path: /data/exp/ARA/[yyyy]/unblinded/L1/ARA0[Station]/[mmdd]/run[xxxxxx]/

    If it is data,
    Usage = python3 %s
    <Station ex) 2>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    station=str(sys.argv[1])
    output=str(sys.argv[2])

    config_loader(CPath = curr_path+'/..', Station = station, Output = output)

del curr_path
