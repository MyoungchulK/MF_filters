import sys
import numpy as np
import re
from glob import glob
from datetime import datetime
from subprocess import call

def bin_range_maker(data,data_len):

    data_bins = np.linspace(data[0], data[-1], data_len + 1)
    data_bin_center = (data_bins[1:] + data_bins[:-1]) * 0.5

    return data_bins, data_bin_center

def file_sorter(d_path):

    # data path
    d_list_chaos = glob(d_path)
    d_len = len(d_list_chaos)
    print('Total Runs:',d_len)

    # make run list
    run_tot=np.full((d_len),-1,dtype=int)
    aa = 0
    for d in d_list_chaos:
        run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
        aa += 1
    del aa

    # sort the run and path
    run_index = np.argsort(run_tot)
    run_tot = run_tot[run_index]
    d_list = []
    for d in range(d_len):
        d_list.append(d_list_chaos[run_index[d]])
    del d_list_chaos, d_len, run_index

    run_range = np.arange(run_tot[0],run_tot[-1]+1)

    return d_list, run_tot, run_range

def evt_num_loader(Station, Run, trig_set = 'all', qual_set = 'all', evt_entry = None, act_evt = None, add_info = None):

    import h5py

    d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/Info_A{Station}_R{Run}.h5'

    hf = h5py.File(d_path, 'r')
    evt_num = hf['evt_num'][:]
    if add_info is not None:  # let say it is onyl working for wf_len (interpolation) for now
        add_tree = hf[add_info][1]
        print(f'{add_info} is loaded. size:{add_tree.shape}')
    else:
        add_tree = None

    if evt_entry is not None:
        evt_entry = np.array([evt_entry])
        act_evt_num = hf['act_evt_num'][:]
        print(f'Selected act event is', act_evt_num[np.where(evt_num == evt_entry)[0][0]])
        if evt_entry in evt_num:
            print(f'Selected event is {evt_entry}')
            if add_info is not None:
                add_tree = add_tree[:,evt_entry]
            else:
                add_tree = None
            return evt_entry, add_tree
        else:
            print('Selected event is not in event tree!')
            sys.exit(1)
    else:
        pass

    if act_evt is not None:
        act_evt_num = hf['act_evt_num'][:]
        act_evt = np.array([act_evt])
        if act_evt in act_evt_num:
            print(f'Selected event is {act_evt}')
            evt_entry = evt_num[act_evt_num == act_evt]
            print(f'Entry for selected event is {evt_entry}')
            if add_info is not None:
                add_tree = add_tree[:,evt_entry]
            else:
                add_tree = None
            return evt_entry, add_tree
        else:
            print('Selected event is not in event tree!')
            sys.exit(1)
        del act_evt_num
    else:
        pass

    trig_num = hf['trig_num'][:]
    qual_num = hf['qual_num_pyroot'][:]
    del hf
    if trig_set != 'all' and qual_set == 'all':
        evt_entry = evt_num[trig_num == trig_set]
    elif trig_set == 'all' and qual_set != 'all':
        evt_entry = evt_num[qual_num == qual_set]
    elif trig_set != 'all' and qual_set != 'all':
        evt_entry = evt_num[(trig_num == trig_set) & (qual_num == qual_set)]  
    elif trig_set == 'all' and qual_set == 'all':
        evt_entry = np.copy(evt_num)
   
    if add_info is not None and len(evt_entry) > 0:  # let say it is onyl working for wf_len (interpolation) for now
        add_tree = add_tree[:, evt_entry]
        print(f'Selected size of {add_info} is {add_tree.shape}')
    else:
        add_tree = None
 
    if len(evt_entry) > 0:    
        print(f'# of selected event is {len(evt_entry)} from {len(evt_num)}')
        return evt_entry, add_tree
    else:
        print('There is no desired WF! End the scripts!')
        print(d_path)
        sys.exit(1)

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

def run_config_file_reader(run_start_path):

    with open(run_start_path,'r') as run_start_file:
        run_start = run_start_file.read()

        run_key = 'Run:'
        empty_run_key = np.nan

        time_key = 'Time:'
        empty_time_key = np.nan

        run_end_key = 'Message'

        run_start_num = context_finder(run_start, run_key, time_key, empty_run_key)
        unix_start = context_finder(run_start, time_key, run_end_key, empty_time_key)
        if not np.isfinite(unix_start):
            date_start = np.nan
        else:
            date_start = datetime.fromtimestamp(unix_start)
            date_start = date_start.strftime('%Y%m%d%H%M%S')
            date_start = int(date_start)
        run_start_info = np.array([run_start_num,unix_start,date_start])
    
    return run_start_info

def config_collector(Data, Station, Run, Year):

    print('Collecting conifg. info starts!')

    run = list('000000')
    run[-len(str(Run)):] = str(Run)
    run = "".join(run)

    if Year != 2013:
        d_path = Data[:-16]
        config_path = f'{d_path}configFile.run{run}.dat'
        run_start_path = f'{d_path}runStart.run{run}.dat'
        run_stop_path = f'{d_path}runStop.run{run}.dat'
    else:
        d_path = f'/data/exp/ARA/{Year}/raw/ARA0{Station}-SPS-ARA/run_{run}/logs/'
        config_path = f'{d_path}configFile.run{run}.dat'
        run_start_path = f'{d_path}runStart.run{run}.dat'
        run_stop_path = f'{d_path}runStop.run{run}.dat' 
   
    try: 
        LS_CMD = f'ls {d_path}'
        call(LS_CMD.split(' '))
    except FileNotFoundError:
        print('Wrong directory!!')
        print(d_path)
        run_start_info = np.full((3),np.nan)      
        run_stop_info = np.full((3),np.nan)      
        masked_ant = np.full((20),np.nan)
        rf_block_num = np.array([np.nan])
        soft_block_num = np.array([np.nan])
        trig_win_num = np.array([np.nan])
        delay_enable = np.array([np.nan])
        delay_num = np.full((20),np.nan)
        month = np.nan
        date = np.nan
 
        print('Everything is NAN!')

        return run_start_info, run_stop_info, masked_ant, rf_block_num, soft_block_num, trig_win_num, delay_enable, delay_num, month, date     
    
    try:
        run_start_info = run_config_file_reader(run_start_path)
    
    except FileNotFoundError:
        print('There is no run start file!!')
        run_start_info = np.full((3),np.nan)

    try:
        run_stop_info = run_config_file_reader(run_stop_path)

    except FileNotFoundError:
        print('There is no run stop file!!')
        run_stop_info = np.full((3),np.nan)

    try:
        with open(config_path,'r') as config_file:
            config = config_file.read()

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

            masked_ant = context_finder(config, mask_key, end_key, empty_l1_mask)
            rf_block_num = np.array([context_finder(config, rf_key, end_key, empty_rf_block)])
            soft_block_num = np.array([context_finder(config, soft_key, end_key, empty_soft_block)])
            trig_win_num = np.array([context_finder(config, trig_key, end_key, empty_trig_win)])
            delay_enable = np.array([context_finder(config, delay_enable_key, end_key, empty_delay_enable)])
            delay_num = context_finder(config, delay_key, end_key, empty_delay)

    except FileNotFoundError:
        print('There is no config. file!!')
        masked_ant = np.full((20),np.nan)
        rf_block_num = np.array([np.nan])
        soft_block_num = np.array([np.nan])
        trig_win_num = np.array([np.nan])
        delay_enable = np.array([np.nan])
        delay_num = np.full((20),np.nan)
  
    if np.isfinite(run_start_info[2]):
        month = int(str(run_start_info[2])[4:6])
        date = int(str(run_start_info[2])[6:8])
    else:
        month = np.nan
        date = np.nan
    
    print('Run start info:',run_start_info) 
    print('Run stop info:',run_stop_info) 
    print('Collecting conifg. info is done!')

    return run_start_info, run_stop_info, masked_ant, rf_block_num, soft_block_num, trig_win_num, delay_enable, delay_num, month, date

def list_maker(glob_path, Station, Year):

    d_list = glob(glob_path)
    d_run_num = []
    for d in d_list:
        run_num = int(re.sub("\D", "", d[-11:]))
        d_run_num.append(run_num)
        del run_num
    d_run_num = np.asarray(d_run_num)

    # sort the run and path
    d_run_idx = np.argsort(d_run_num)
    d_run_num_sort = d_run_num[d_run_idx]
    d_list_sort = []
    for s in range(len(d_run_num_sort)):
        d_list_sort.append(d_list[d_run_idx[s]])
    del d_list, d_run_num, d_run_idx

    if Station ==3 and Year == 2018:
        digit5_run_idx = np.where(d_run_num_sort >= 10000)[0]
        print(digit5_run_idx)
        d_list_sort_2018 = []
        for s in range(len(digit5_run_idx)):
            d_list_sort_2018.append(d_list_sort[digit5_run_idx[s]])
        print(d_run_num_sort)
        d_run_num_sort = d_run_num_sort[digit5_run_idx]
        print(d_run_num_sort)
        d_list_sort = d_list_sort_2018
    else:
        pass

    return d_list_sort, d_run_num_sort

def general_data_ped_list(Station, Year):

    # make data list
    data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/unblinded/L1/ARA0{Station}/[0-9][0-9][0-9][0-9]/run[0-9][0-9][0-9][0-9][0-9][0-9]/event[0-9][0-9][0-9][0-9][0-9][0-9].root', Station, Year)
    #data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/unblinded/L1/ARA0{Station}/*/run*/event[0-9]*.root', Station, Year)

    # make ped list
    ped_list, ped_run = list_maker(f'/data/exp/ARA/{Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, Year)

    # make last year ped list
    last_Year = int(Year-1)
    if last_Year == 2013:
        last_ped_list, last_ped_run = list_maker(f'/data/user/mkim/ARA_{last_Year}_Ped/ARA0{Station}/*pedestalValues*', Station, last_Year)
    elif Year == 2018 and Station == 3:
        last_Year = int(Year-2)
        last_ped_list, last_ped_run = list_maker(f'/data/exp/ARA/{last_Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, last_Year)
    else:
        last_ped_list, last_ped_run = list_maker(f'/data/exp/ARA/{last_Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, last_Year)

    # total ped list
    ped_list = last_ped_list + ped_list
    ped_run = np.append(last_ped_run, ped_run, axis=0)

    return data_list, data_run, ped_list, ped_run

def A235_data_ped_list(Station, Year):

    if Year == 2013 and Station != 5: #when we were full of hope and dream

        # make data list
        data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/filtered/burnSample1in10/ARA0{Station}/root/*/event[0-9]*.root', Station, Year)

        # total ped list
        ped_list, ped_run = list_maker(f'/data/user/mkim/ARA_2013_Ped/ARA0{Station}/*pedestalValues*', Station, Year)

    elif Year > 2013 and Year < 2017 and Station != 5:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    elif Year == 2017 and Station == 2:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    elif Year == 2018:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    else:

        print('Wrong Station & Year combination!')
        print('Choose 1) 2013~2016:ARA2&3, 2) 2017:ARA2, 3) 2018:ARA2&3&5')
        sys.exit(1)

    return data_list, data_run, ped_list, ped_run

def dag_statement(r, data_list, ped_list, Station, data_run):

    contents = ""
    contents += f'JOB job_{r} ARA_job.sub \n'
    #contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" out="{Output}" station="{Station}" run="{data_run}"\n\n'
    contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" station="{Station}" run="{data_run}"\n\n'

    return contents

def data_info_reader(d_path_str):

    # it must be start with /data/exp/ARA/.....
    # these informations might can salvage from root file itself in future...

    # salvage just number
    d_path = re.sub("\D", "", d_path_str) 

    # year
    yr = int(d_path[:4])
    if yr == 2013:

        # station 
        st = int(d_path[7:9])

        # run 
        run = int(re.sub("\D", "", d_path_str[-11:]))

        # config
        config = config_checker(st, run)

        # month
        m = -1

        # date
        d = -1
 
        print(f'data info. 1)station:{st} 2)year:{yr} 3)month:{m} 4)date:{d} 5)run:{run} 6)config:{config}')

    else:

        # station
        st = int(d_path[5:7])

        # run
        run = int(d_path[11:17])

        # config
        config = config_checker(st, run)

        # month
        m = int(d_path[7:9])
        
        # date
        d = int(d_path[9:11])

        print(f'data info. 1)station:{st} 2)year:{yr} 3)month:{m} 4)date:{d} 5)run:{run} 6)config:{config}')

    return st, run, config, yr, m, d

def config_checker(st, runNum):

    # from Brian: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_Cuts.h

    # default. unknown
    config=0

    # by statation
    if st == 2:
        if runNum>=0 and runNum<=4:
            config=1
        elif runNum>=11 and runNum<=60:
            config=4
        elif runNum>=120 and runNum<=2274:
            config=2
        elif runNum>=2275 and runNum<=3463:
            config=1
        elif runNum>=3465 and runNum<=4027:
            config=3
        elif runNum>=4029 and runNum<=6481:
            config=4
        elif runNum>=6500 and runNum<=8097:
            config=5
        #elif runNum>=8100 and runNum<=8246:
        elif runNum>=8100 and runNum<=9504:
            config=4
        elif runNum>=9505 and runNum<=9748:
            config=5
        elif runNum>=9749:
            config=6
        else:
            pass

    elif st == 3:
        if runNum>=0 and runNum<=4:
            config=1
        elif runNum>=470 and runNum<=1448:
            config=2
        elif runNum>=1449 and runNum<=1901:
            config=1
        elif runNum>=1902 and runNum<=3103:
            config=5
        elif runNum>=3104 and runNum<=6004:
            config=3
        elif runNum>=6005 and runNum<=7653:
            config=4
        elif runNum>=7658 and runNum<=7808:
            config=3
        elif runNum>=10001:
            config=6
        else:
            pass

    elif st == 5:
        pass

    return config

