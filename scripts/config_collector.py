##
# @file config_collector.py
#
# @section Created on 08/03/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to collect data from configFile.run[xxxxxx].dat, runStart.run[xxxxxx].dat, and runStop.run[xxxxxx].dat

import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
from datetime import datetime
import h5py

def context_finder(txt_read, key, end_key, empty_format = np.nan):
    """! function for scrap information between two string variables. It will find the string between the variables and convert into numpy array

    @param txt_read  
    @param key  string
    @param end_key string
    @param empty_format float
    @return val integer
    """

    key_idx = txt_read.find(key) # find the index of the key
    if key_idx != -1: # check whether key is in the txt_read or not
        key_idx += len(key)
        end_key_idx = txt_read.find(end_key, key_idx) # find the end_key index after key_idx

        if key == 'enableL1Trigger#I20=' or key == 'triggerDelays#I16=': # multiple elements
            val = np.asarray(txt_read[key_idx:end_key_idx].split(","), dtype = int)
        else:
            val = int(txt_read[key_idx:end_key_idx])
    else:
        val = empty_format # it there is no key in the txt_read, output numpy nan

    return val

def scrap_run_dat_info(run_list, run_key, time_key, run_end_key):
    """! function for scraping the run number, and unix time from run dat file and convert into UTC date time

    @param run_list  string. dat file path
    @param run_key  string  
    @param time_key  string
    @param run_end_key  string
    @return run_num  integer
    @return unix_time integer
    @return date_time integer
    """

    ## open dat file and scrap the information
    with open(run_list,'r') as run_file:
        run_read = run_file.read()

        ## scrap the information from context
        ## check whether output of context_finder has a information or not
        ## some of files dont have a all the information...
        
        run_num = context_finder(run_read, run_key, time_key)
        if not np.isfinite(run_num):
            print(f'{run_list} doesnt has run number!! Use file name for identifying run number!')
            run_num = int(run_list[-10:-4]) # I dont like it...
        
        unix_time = context_finder(run_read, time_key, run_end_key)
        if not np.isfinite(unix_time):
            print(f'{run_list} doesnt has unix time!! Output date time as a NAN...')
            date_time = np.nan
        else:
            date_time = datetime.utcfromtimestamp(unix_time)
            date_time = date_time.strftime('%Y%m%d%H%M%S')
            date_time = int(date_time) # save date time as an interger for convenience
    
    return run_num, unix_time, date_time

def main(Station, Output):
    """! main function for collecting configuration

    @param Station  integer. station id ex) 1 ~ 5. not for PA at this moment
    @param Output  string. desired path for saving output
    """

    print('Collectiong file paths from each year!')
    ## all the file name variables
    config_dat_str = 'configFile.run[0-9]*.dat'
    run_start_str = 'runStart.run[0-9]*.dat'
    run_stop_str = 'runStop.run[0-9]*.dat'

    ## list for all file path
    config_list = [] # saving all configFile.run[xxxxxx].dat
    run_start_list = [] # saving for all runStart.run[xxxxxx].dat
    run_stop_list = [] # savinf for all runStop.run[xxxxxx].dat

    ## loop over all L1 data path. 2013 ~ 2019
    for y in tqdm(range(2013, 2019 + 1)):
        if int(y) == 2013: # 2013 has different file path 
            #g_path = f'/data/exp/ARA/{int(y)}/filtered/unzippedTGZFiles/ARA0{Station}/run_[0-9]*/logs/' # alternate path
            g_path = f'/data/exp/ARA/{int(y)}/raw/ARA0{Station}-SPS-ARA/run_[0-9]*/logs/'
        else:
            g_path = f'/data/exp/ARA/{int(y)}/unblinded/L1/ARA0{Station}/*/run*/'
        config_list += glob(f'{g_path}{config_dat_str}')
        run_start_list += glob(f'{g_path}{run_start_str}')
        run_stop_list += glob(f'{g_path}{run_stop_str}')

    config_len = len(config_list)
    run_start_len = len(run_start_list)
    run_stop_len = len(run_stop_list)
    print('Total # of config files:',config_len)
    print('Total # of runstart files:',run_start_len)
    print('Total # of runstop files:',run_stop_len)

    print('Collecting information. 3 for loop')
    ## numpy array pad for storing all the configuration from each run
    run_num = np.arange(20000, dtype = int)
    run_num_len = len(run_num) 
    unix_time = np.full((2, run_num_len), np.nan, dtype = float) # unix time stored in runStart.run[xxxxxx].dat and runStop.run[xxxxxx].dat
    date_time = np.copy(unix_time) # UTC date time converted from unix time
    rf_block_num = np.full((run_num_len), np.nan, dtype = float) # number of RF blocks in each run
    soft_block_num = np.copy(rf_block_num) # number of Software block number in each run
    trig_win_num = np.copy(rf_block_num) # number trigger window length in each run
    delay_enable  = np.copy(rf_block_num) # trigger delay enable. 1: enable, 0: not enable
    delay_num = np.full((16, run_num_len), np.nan, dtype = float) # number of trigger delay in each run
    masked_ant = np.full((20, run_num_len), np.nan, dtype = float) # masked antenna. 1: unmasked, 0: masked

    ## 3 each 'for loop' for collecting information. Since not all the runs have a config, runstart, or runstop, we need to scrap the information from each list      
    ## variables for searching config information
    run_key = 'Run:'
    time_key = 'Time:'
    run_end_key = 'Message'
    masked_ant_key = 'enableL1Trigger#I20='
    rf_blk_key = 'numRF0TriggerBlocks#1='
    soft_blk_key = 'numSoftTriggerBlocks#1='
    trig_win_key = 'triggerWindowSize#1='
    delay_enable_key = 'enableTriggerDelays#I1='
    delay_key = 'triggerDelays#I16='
    end_key = ';'
    empty_masked_ant_foramt = np.full((20), np.nan, dtype = float)
    empty_delay_format = np.full((16), np.nan ,dtype = float)

    ## 1st run start files
    for runs in tqdm(range(run_start_len)):
        run_start, unix_start, date_start = scrap_run_dat_info(run_start_list[runs], run_key, time_key, run_end_key)
        unix_time[0, run_start] = unix_start
        date_time[0, run_start] = date_start
 
    ## 2nd run stop files. I believe if the run is suddenly terminated, that run usually dont have a runstop files 
    for runs in tqdm(range(run_stop_len)):
        run_stop, unix_stop, date_stop = scrap_run_dat_info(run_stop_list[runs], run_key, time_key, run_end_key)
        unix_time[1, run_stop] = unix_stop
        date_time[1, run_stop] = date_stop
 
    ## 3rd config files
    for runs in tqdm(range(config_len)):
        run_config = int(config_list[runs][-10:-4]) # scrap the run number from config file path

        ## open the each config file and scrap the information by context_finder function
        with open(config_list[runs],'r') as config_file:
            config_read = config_file.read()

            rf_block_num[run_config] = context_finder(config_read, rf_blk_key, end_key)
            soft_block_num[run_config] = context_finder(config_read, soft_blk_key, end_key)
            trig_win_num[run_config] = context_finder(config_read, trig_win_key, end_key)
            delay_enable[run_config] = context_finder(config_read, delay_enable_key, end_key)
            delay_num[:, run_config] = context_finder(config_read, delay_key, end_key, empty_format = empty_delay_format)
            masked_ant[:, run_config] = context_finder(config_read, masked_ant_key, end_key, empty_format = empty_masked_ant_foramt)        

    print('Saving information')
    ## create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)

    ## save into h5 file
    h5_file_name=f'Config_A{Station}.h5'
    hf = h5py.File(h5_file_name, 'w')
    hf.create_dataset('run_num', data=run_num, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('date_time', data=date_time, compression="gzip", compression_opts=9)
    hf.create_dataset('masked_ant', data=masked_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('rf_block_num', data=rf_block_num, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_block_num', data=soft_block_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_win_num', data=trig_win_num, compression="gzip", compression_opts=9)
    hf.create_dataset('delay_enable', data=delay_enable, compression="gzip", compression_opts=9)
    hf.create_dataset('delay_num', data=delay_num, compression="gzip", compression_opts=9)
    hf.close() 

    print(f'output is {Output}{h5_file_name}')
    print('Done!')

if __name__ == "__main__":

    if len (sys.argv) != 3:
        Usage = """
    This is designed to collect data from configFile.run[xxxxxx].dat, runStart.run[xxxxxx].dat, and runStop.run[xxxxxx].dat
    Usage = python3 %s
    <Station ex) 2>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    ## argv
    station=str(sys.argv[1])
    output=str(sys.argv[2])
    print("Station {}, Output {}".format(station, output))

    ## excute main function
    main(Station = station, Output = output)
