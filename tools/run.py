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

def bad_run_checker(run, st):

    bad_run_bool = False

    if run in bad_surface_run(st):
        bad_run_bool = True
        print(f'This run{run}, station{st} is flagged as a bad surface run!')    

    elif run in bad_run(st):
        bad_run_bool = True  
        print(f'This run{run}, station{st} is flagged as a bad run!')

    else:
        pass

    return bad_run_bool

def bad_surface_run(st):

    # masked run(2014~2016) from brian's analysis
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L782

    # array for bad run
    bad_run = np.array([], dtype=int)

    if st == 2:

        # Runs shared with Ming-Yuan
        # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889

        bad_run = np.append(bad_run, 2090)
        bad_run = np.append(bad_run, 2678)
        bad_run = np.append(bad_run, 4777)
        bad_run = np.append(bad_run, 5516)
        bad_run = np.append(bad_run, 5619)
        bad_run = np.append(bad_run, 5649)
        bad_run = np.append(bad_run, 5664)
        bad_run = np.append(bad_run, 5666)
        bad_run = np.append(bad_run, 5670)
        bad_run = np.append(bad_run, 5680)
        bad_run = np.append(bad_run, 6445)
        bad_run = np.append(bad_run, 6536)
        bad_run = np.append(bad_run, 6542)
        bad_run = np.append(bad_run, 6635)
        bad_run = np.append(bad_run, 6655)
        bad_run = np.append(bad_run, 6669)
        bad_run = np.append(bad_run, 6733)

        # Runs identified independently

        bad_run = np.append(bad_run, 2091)
        bad_run = np.append(bad_run, 2155)
        bad_run = np.append(bad_run, 2636)
        bad_run = np.append(bad_run, 2662)
        bad_run = np.append(bad_run, 2784)
        bad_run = np.append(bad_run, 4837)
        bad_run = np.append(bad_run, 4842)
        bad_run = np.append(bad_run, 5675)
        bad_run = np.append(bad_run, 5702)
        bad_run = np.append(bad_run, 6554)
        bad_run = np.append(bad_run, 6818)
        bad_run = np.append(bad_run, 6705)
        bad_run = np.append(bad_run, 8074)
       
    elif st == 3:

        # Runs shared with Ming-Yuan
        # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2041

        bad_run = np.append(bad_run, 977)
        bad_run = np.append(bad_run, 1240)
        bad_run = np.append(bad_run, 3158)
        bad_run = np.append(bad_run, 3431)
        bad_run = np.append(bad_run, 3432)
        bad_run = np.append(bad_run, 3435)
        bad_run = np.append(bad_run, 3437)
        bad_run = np.append(bad_run, 3438)
        bad_run = np.append(bad_run, 3439)
        bad_run = np.append(bad_run, 3440)
        bad_run = np.append(bad_run, 3651)
        bad_run = np.append(bad_run, 3841)
        bad_run = np.append(bad_run, 4472)
        bad_run = np.append(bad_run, 4963)
        bad_run = np.append(bad_run, 4988)
        bad_run = np.append(bad_run, 4989)

        # Runs identified independently

        bad_run = np.append(bad_run, 1745)
        bad_run = np.append(bad_run, 3157)
        bad_run = np.append(bad_run, 3652)
        bad_run = np.append(bad_run, 3800)
        bad_run = np.append(bad_run, 6193)
        bad_run = np.append(bad_run, 6319)
        bad_run = np.append(bad_run, 6426)

        # Runs I am sure we will exclude...

        bad_run = np.append(bad_run, 2000)
        bad_run = np.append(bad_run, 2001)
        
    else:
        pass

    return bad_run

def bad_run(st):

    # masked run(2014~2016) from brian's analysis 
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L881
    
    # array for bad run
    bad_run = np.array([], dtype=int)

    if st == 2:

        ## 2013 ##

        ## 2014 ##
        # 2014 rooftop pulsing, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, [3120, 3242])

        # 2014 surface pulsing
        # originally flagged by 2884, 2895, 2903, 2912, 2916
        # going to throw all runs jan 14-20
        bad_run = np.append(bad_run, 2884) # jan 14 2014 surface pulser runs. actual problem causer
        bad_run = np.append(bad_run, [2885, 2889, 2890, 2891, 2893]) # exclusion by proximity
        
        bad_run = np.append(bad_run, 2895) # jan 16 2014 surface pulser runs. actual problem causer
        bad_run = np.append(bad_run, 2898) # exclusion by proximity
        bad_run = np.append(bad_run, [2900, 2901, 2902]) # jan 17 2014. exclusion by proximity
        
        bad_run = np.append(bad_run, 2903) # # jan 18 2014 surface pulser runs. actual problem causer
        bad_run = np.append(bad_run, [2905, 2906, 2907]) # exclusion by proximity
        
        bad_run = np.append(bad_run, 2912) # # jan 19 2014 surface pulser runs. actual problem causer
        bad_run = np.append(bad_run, 2915) # exclusion by proximity
        
        bad_run = np.append(bad_run, 2916) # jan 20 2014 surface pulser runs. actual problem causer
        bad_run = np.append(bad_run, 2918) # exclusion by proximity

        # surface pulsing from m richman (identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14)
        bad_run = np.append(bad_run, [2938, 2939])

        # 2014 Cal pulser sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, np.arange(3139, 3162+1))
        bad_run = np.append(bad_run, np.arange(3164, 3187+1))
        bad_run = np.append(bad_run, np.arange(3289, 3312+1))

        """
        # ARA02 stopped sending data to radproc. Alert emails sent by radproc.
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        # http://ara.icecube.wisc.edu/wiki/index.php/Drop_29_3_2014_ara02
        bad_run = np.append(bad_run, 3336)
        """

        # 2014 L2 Scaler Masking Issue. 
        # Cal pulsers sysemtatically do not reconstruct correctly, rate is only 1 Hz
        # Excluded because configuration was not "science good"
        bad_run = np.append(bad_run, np.arange(3464, 3504+1))

        # 2014 Trigger Length Window Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, np.arange(3578, 3598+1))

        """
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        # 2014, 4th June, Checking the functionality of the L1Scaler mask. 
        bad_run = np.append(bad_run, 3695) # Masiking Ch0,1, 14
        bad_run = np.append(bad_run, 3700) # Masiking Ch2, 14
        bad_run = np.append(bad_run, 3701) # Masiking Ch4,5, 14
        bad_run = np.append(bad_run, 3702) # Masiking Ch6,7, 14
        bad_run = np.append(bad_run, 3703) # Masiking Ch8,9, 14
        bad_run = np.append(bad_run, 3704) # Masiking Ch10,11, 14
        bad_run = np.append(bad_run, 3705) # Masiking Ch12,13, 14
        bad_run = np.append(bad_run, 3706) # Masiking Ch14, 15

        # 2014, 16th June, Software update on ARA02 to fix the L1triggers.
        bad_run = np.append(bad_run, 3768)

        # 2014, 31st July, Testing new software to change trigger and readout window, pre-trigger samples.
        bad_run = np.append(bad_run, np.arange(3988, 3994+1))

        # 2014, 5th Aug, More tests on the pre-trigger samples.
        bad_run = np.append(bad_run, np.arange(4019, 4022+1))

        # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
        bad_run = np.append(bad_run, 4029)

        # 2014, 14th Aug, Finally changed trigger window size to 170ns.
        # http://ara.icecube.wisc.edu/wiki/index.php/File:Gmail_-_-Ara-c-_ARA_Operations_Meeting_Tomorrow_at_0900_CDT.pdf
        bad_run = np.append(bad_run, 4069)
        """

        ## 2015 ##
        # ??
        bad_run = np.append(bad_run, 4004)

        # 2015 icecube deep pulsing
        # 4787 is the "planned" run
        # 4795,4797-4800 were accidental
        bad_run = np.append(bad_run, 4785) # accidental deep pulser run (http://ara.physics.wisc.edu/docs/0017/001719/003/181001_ARA02AnalysisUpdate.pdf, slide 38)
        bad_run = np.append(bad_run, 4787) # deep pulser run (http://ara.physics.wisc.edu/docs/0017/001724/004/181015_ARA02AnalysisUpdate.pdf, slide 29)
        bad_run = np.append(bad_run, np.arange(4795, 4800+1))

        # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
        bad_run = np.append(bad_run, np.arange(4820, 4825+1))
        bad_run = np.append(bad_run, np.arange(4850, 4854+1))
        bad_run = np.append(bad_run, np.arange(4879, 4936+1))
        bad_run = np.append(bad_run, np.arange(5210, 5277+1))

        # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
        bad_run = np.append(bad_run, [4872, 4873])
        bad_run = np.append(bad_run, 4876) # Identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14

        # 2015 Pulser Lift, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 2)
        # Run number from private communication with John Kelley
        bad_run = np.append(bad_run, 6513) 

        # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
        bad_run = np.append(bad_run, 6527)

        ## 2016 ##
        """
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
        # 2016, 21st July, Reduced trigger delay by 100ns.
        bad_run = np.append(bad_run, 7623)
        """

        # 2016 cal pulser sweep, Jan 2015?, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
        bad_run = np.append(bad_run, np.arange(7625, 7686+1)) 

        ## other ##
        # D1 Glitches, Identified by MYL as having glitches after long periods of downtime
        bad_run = np.append(bad_run, 3)
        bad_run = np.append(bad_run, 11)
        bad_run = np.append(bad_run, 59)
        bad_run = np.append(bad_run, 60)
        bad_run = np.append(bad_run, 71)

        # Badly misreconstructing runs
        # run 8100. Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015. 
        bad_run = np.append(bad_run, np.arange(8100, 8246+1))

        ## 2017 ## 
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2017
        # 01/16/2017, Rooftop pulser run, Hpol ran for 30 min at 1 Hz starting 22:13:06. Vpol ran for 30 min at 1 Hz starting 22:44:50.
        bad_run = np.append(bad_run, 8530)

        # 01/24/2017, Deep pulser run, IC string 1 shallow pulser ~23:48-00:00. IC string 22 shallow pulser (Jan 25) ~00:01-00:19.
        bad_run = np.append(bad_run, 8573)

        # 01/25/2017, A2D6 pulser lift, Ran in continuous noise mode with V&Hpol Tx.
        bad_run = np.append(bad_run, [8574, 8575])

        # 01/25/2017, Same configuration as run8575, Ran in continuous noise mode with Hpol Tx. Forgot to switch back to normal configuration. No pulser lift in this period.
        bad_run = np.append(bad_run, [8576, 8578])

        # Cal pulser attenuation sweep
        """
        bad_run = np.append(bad_run, 8953) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
        bad_run = np.append(bad_run, np.arange(8955, 8956+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
        bad_run = np.append(bad_run, np.arange(8958, 8962+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
        """
        bad_run = np.append(bad_run, np.arange(8963, 9053+1)) # 04/10/2017, System crashed on D5 (D6 completed successfully); D6 VPol 0 dB is 8963...D6 VPol 31 dB is 8974...D6 HPol 0 dB is 8975...D5 VPol 0 dB is 9007...crashed before D5 HPol
        bad_run = np.append(bad_run, np.arange(9129, 9160+1)) # 04/25/2017, D6 VPol: 9129 is 0 dB, 9130 is 1 dB, ... , 9160 is 31 dB
        bad_run = np.append(bad_run, np.arange(9185, 9216+1)) # 05/01/2017, D6 HPol: 9185 is 0 dB, 9186 is 1 dB, ... , 9216 is 31 dB
        bad_run = np.append(bad_run, np.arange(9231, 9262+1)) # 05/04/2017, D5 VPol: 9231 is 0 dB, ... , 9262 is 31 dB
        bad_run = np.append(bad_run, np.arange(9267, 9298+1)) # 05/05/2017, D5 HPol: 9267 is 0 dB, ... , 9298 is 31 dB

        ## 2018 ##
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
        # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

        ## 2019 ##
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2019
        # D5 Calpulser sweep, 01/25/2019
        bad_run = np.append(bad_run, np.arange(12842, 12873+1)) # D5 Vpol attenuation sweep 0 to 31 dB with a step of 1 dB.
        bad_run = np.append(bad_run, np.arange(12874, 12905+1)) # D5 Hpol attenuation sweep 0 to 31 dB with a step of 1 dB. Wanted to verify if D5 Hpol actually fires or not. Conclusion was that D5 Hpol does not fire and ARA02 defaults to firing D5 Vpol instead.
        
        # D6 Vpol fired at 0 dB attenuation. Trigger delays of ARA2 ch. adjusted.
        # 03/22/2019 ~ 04/11/2019 
        bad_run = np.append(bad_run, np.arange(13449, 13454+1))
        bad_run = np.append(bad_run, np.arange(13455, 13460+1))
        bad_run = np.append(bad_run, np.arange(13516, 13521+1))
        bad_run = np.append(bad_run, np.arange(13522, 13527+1))
        bad_run = np.append(bad_run, np.arange(13528, 13533+1))
        bad_run = np.append(bad_run, 13542)
        bad_run = np.append(bad_run, np.arange(13543, 13547+1))
        bad_run = np.append(bad_run, 13549)
        bad_run = np.append(bad_run, np.arange(13550, 13554+1))
        bad_run = np.append(bad_run, np.arange(13591, 13600+1))
        bad_run = np.append(bad_run, np.arange(13614, 13628+1))
        bad_run = np.append(bad_run, np.arange(13630, 13644+1))
        bad_run = np.append(bad_run, np.arange(13654, 13663+1))
        bad_run = np.append(bad_run, np.arange(13708, 13723+1))
        bad_run = np.append(bad_run, np.arange(13732, 13746+1))
        bad_run = np.append(bad_run, np.arange(13757, 13771+1))
        bad_run = np.append(bad_run, np.arange(13772, 13775+1))

        # Trigger delays of ARA2 ch. 
        # 04/18/2019 ~ 05/2/2019
        bad_run = np.append(bad_run, np.arange(13850, 13875+1))
        bad_run = np.append(bad_run, np.arange(13897, 13898+1))
        bad_run = np.append(bad_run, np.arange(13900, 13927+1))
        bad_run = np.append(bad_run, np.arange(13967, 13968+1))
        bad_run = np.append(bad_run, np.arange(13970, 13980+1))
        bad_run = np.append(bad_run, np.arange(13990, 14004+1))
        bad_run = np.append(bad_run, np.arange(14013, 14038+1))
        bad_run = np.append(bad_run, np.arange(14049, 14053+1))
        bad_run = np.append(bad_run, np.arange(14055, 14060+1))
        bad_run = np.append(bad_run, np.arange(14079, 14087+1))
        bad_run = np.append(bad_run, np.arange(14097, 14105+1))
        bad_run = np.append(bad_run, np.arange(14115, 14123+1))
        bad_run = np.append(bad_run, np.arange(14133, 14141+1))
        bad_run = np.append(bad_run, np.arange(14160, 14185+1))
        bad_run = np.append(bad_run, np.arange(14194, 14219+1))
        bad_run = np.append(bad_run, np.arange(14229, 14237+1))

        # need more investigation
        bad_run = np.append(bad_run, 4829) 
        bad_run = np.append(bad_run, [8562, 8563, 8567, 8568, 8572])
        bad_run = np.append(bad_run, 8577)
        bad_run = np.append(bad_run, [9748, 9750])
        bad_run = np.append(bad_run, np.arange(9522, 9849))

        # short run
        bad_run = np.append(bad_run, 6480)
        bad_run = np.append(bad_run, 10125)

    elif st == 3:

        ## 2013 ##
        # Misc tests: http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2013
        # bad_run = np.append(bad_run, np.arange(22, 62+1))      

        # ICL rooftop: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
        # bad_run = np.append(bad_run, np.arange(63, 70+1))
        # bad_run = np.append(bad_run, np.arange(333, 341+1))

        # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
        # bad_run = np.append(bad_run, np.arange(72, 297+1))
        # bad_run = np.append(bad_run, np.arange(346, 473+1))
 
        # Eliminate all early data taking (all runs before 508)
        bad_run = np.append(bad_run, np.arange(508+1))

        # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
        # ??

        ## 2014 ##
        # 2014 Rooftop Pulser, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, [2235, 2328])

        # 2014 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, np.arange(2251, 2274+1))
        bad_run = np.append(bad_run, np.arange(2376, 2399+1))

        """
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
        bad_run = np.append(bad_run, 3063)
   
        # 2014, 14th Aug, Finally changed trigger window size to 170ns.
        bad_run = np.append(bad_run, 3103)
        """

        ## 2015 ##
        # 2015 surface or deep pulsing
        # got through cuts
        # happened jan 5-6, some jan 8
        # waveforms clearly show double pulses or things consistent with surface pulsing
        bad_run = np.append(bad_run, 3811)        
        bad_run = np.append(bad_run, [3810, 3820, 3821, 3822]) # elminated by proximity to deep pulser run
        bad_run = np.append(bad_run, 3823) # deep pulser, observation of 10% iterator event numbers 496, 518, 674, 985, 1729, 2411

        # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
        bad_run = np.append(bad_run, np.arange(3844, 3860+1))       
        bad_run = np.append(bad_run, np.arange(3881, 3891+1))       
        bad_run = np.append(bad_run, np.arange(3916, 3918+1))       
        bad_run = np.append(bad_run, np.arange(3920, 3975+1))       
        bad_run = np.append(bad_run, np.arange(4009, 4073+1)) 

        # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
        bad_run = np.append(bad_run, [3977, 3978])   
       
        # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
        bad_run = np.append(bad_run, 6041)

        # 2015 station anomaly
        # see moni report: http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1213
        # identified by MYL: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
        bad_run = np.append(bad_run, np.arange(4914, 4960+1))   

        ## 2016 ##

        """
        http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
        # 2016, 21st July, Reduced trigger delay by 100ns.
        bad_run = np.append(bad_run, 7124)
        """

        # More events with no RF/deep triggers, seems to precede coming test
        bad_run = np.append(bad_run, 7125)

        # 2016 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, np.arange(7126, 7253+1))

        """
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
        # 2016 Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015.
        bad_run = np.append(bad_run, 7658)
        """

        ## 2018 ##
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
        # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

        # need more investigation
        bad_run = np.append(bad_run, np.arange(12788, 12832))
        bad_run = np.append(bad_run, np.arange(12866, 13087))

        # short run
        bad_run = np.append(bad_run, 1125)
        bad_run = np.append(bad_run, 1126)
        bad_run = np.append(bad_run, 1129)
        bad_run = np.append(bad_run, 1130)
        bad_run = np.append(bad_run, 1132)
        bad_run = np.append(bad_run, 1133)
        bad_run = np.append(bad_run, 1139)
        bad_run = np.append(bad_run, 1140)
        bad_run = np.append(bad_run, 1141)
        bad_run = np.append(bad_run, 1143)
        bad_run = np.append(bad_run, 10025)
        bad_run = np.append(bad_run, 10055)
        bad_run = np.append(bad_run, 11333)
        bad_run = np.append(bad_run, 11418)
        bad_run = np.append(bad_run, 11419)
        bad_run = np.append(bad_run, 12252)
        bad_run = np.append(bad_run, 12681)
        bad_run = np.append(bad_run, 12738)
    
    elif st == 5:

        ## 2018 ##
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
        # Calibration pulser lowered, http://ara.physics.wisc.edu/docs/0015/001589/002/ARA5CalPulser-drop-Jan-2018.xlsx

        # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx
        bad_run = np.append(bad_run)
    
    else:
        pass

    return bad_run

def bad_unixtime(st, unix_time):

    # masked unixtime(2014~2016) from brian's analysis
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L503

    bad_unit_t = False

    if st == 2:

        # Livetime flagged as bad by Biran
        if((unix_time>=1389381600 and unix_time<=1389384000) or # from run 2868
        (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
        # (unix_time>=1449189600 and unix_time<=1449190200) or # from run 6507
        (unix_time>=1449187200 and unix_time<=1449196200) or # from run 6507

        #Livetime flagged as bad by Biran's undergrads
        #config 1
        # (unix_time>=1380234000 and unix_time<=1380236400) or # from run 2428 22 hour balloon launch
        # (unix_time>=1382046000 and unix_time<=1382047500) or # from run 2536 22 hour balloon launch
        (unix_time>=1382712900 and unix_time<=1382713500) or # from run 2575
        (unix_time>=1382972700 and unix_time<=1382973300) or # from run 2589
        # (unix_time>=1383689100 and unix_time<=1383690900) or # from run 2631 22 hour balloon launch
        (unix_time>=1383884400 and unix_time<=1383886200) or # from run 2642
        (unix_time>=1384060200 and unix_time<=1384061100) or # from run 2652
        (unix_time>=1384487400 and unix_time<=1384489800) or # from run 2677
        (unix_time>=1384489980 and unix_time<=1384491060) or # from run 2678 at start may be glitch or continued from 2677
        (unix_time>=1384856520 and unix_time<=1384856640) or # from run 2698 super zoomed in two minute window
        # (unix_time>=1385674200 and unix_time<=1385675100) or # from run 2744 22 hour balloon launch
        (unix_time>=1389381600 and unix_time<=1389383700) or # from run 2868 first of two from run 2868
        (unix_time>=1389398700 and unix_time<=1389400200) or # from run 2868 second of two from run 2868
        (unix_time>=1389665100 and unix_time<=1389666300) or # from run 2884
        (unix_time>=1393288800 and unix_time<=1393289400) or # from run 3099
        # (unix_time>=1397856600 and unix_time<=1397858400) or # from run 3442 22 hour balloon launch

        #config 2
        (unix_time>=1376731800 and unix_time<=1376733000) or # from run 2235

        #conifg 3
        (unix_time>=1400276700 and unix_time<=1400277300) or # from run 3605 mainly looks like glitch at end

        #config 4
        (unix_time>=1409986500 and unix_time<=1409988000) or # from run 4184
        # (unix_time>=1412026200 and unix_time<=1412027100) or # from run 4301 22 hr balloon
        # (unix_time>=1412285400 and unix_time<=1412288100) or # from run 4316 weird 22hr balloon
        # (unix_time>=1412544600 and unix_time<=1412545500) or # from run 4331 22hr balloon
        # (unix_time>=1412803800 and unix_time<=1412804700) or # from run 4346 22hr balloon
        (unix_time>=1413898200 and unix_time<=1413899100) or # from run 4408
        (unix_time>=1414083900 and unix_time<=1414086000) or # from run 4418
        (unix_time>=1414350300 and unix_time<=1414351200) or # from run 4434 pt 1
        # (unix_time>=1414358700 and unix_time<=1414359900) or # from run 4434 pt 2 22hr balloon
        (unix_time>=1414674300 and unix_time<=1414674780) or # from run 4452
        (unix_time>=1414986600 and unix_time<=1414987200) or # from run 4471
        (unix_time>=1415223000 and unix_time<=1415223900) or # from run 4483
        (unix_time>=1415380500 and unix_time<=1415381400) or # from run 4493
        (unix_time>=1415558100 and unix_time<=1415559000) or # from run 4503
        (unix_time>=1415742300 and unix_time<=1415743800) or # from run 4513
        (unix_time>=1416207000 and unix_time<=1416212100) or # from run 4541
        (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
        (unix_time>=1416905100 and unix_time<=1416910500) or # from run 4579 two spikes about an hour apart
        # (unix_time>=1416950700 and unix_time<=1416951600) or # from run 4582 22 hour balloon launch
        (unix_time>=1417677000 and unix_time<=1417678200) or # from run 4621  weird and cool
        (unix_time>=1417836000 and unix_time<=1417837500) or # from run 4631
        (unix_time>=1420097100 and unix_time<=1420098300) or # from run 4763
        (unix_time>=1420293300 and unix_time<=1420294200) or # from run 4774
        (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
        (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
        (unix_time>=1421024400 and unix_time<=1421025300) or # from run 4817
        (unix_time>=1421713200 and unix_time<=1421718600) or # from run 4872 looks full of errors and not spiky but could have a spiky
        (unix_time>=1421718000 and unix_time<=1421725800) or # from run 4873 definitely an error but also has spiky boy, part 1 of 2
        (unix_time>=1421733300 and unix_time<=1421733900) or # from run 4873 spiky boy alone but in a run with errors, part 2 of 2
        (unix_time>=1421783400 and unix_time<=1421794200) or # from run 4876 definitely an error but not a spikey boy
        # (unix_time>=1428529800 and unix_time<=1428530700) or # from run 5389 22 hour balloon launch
        (unix_time>=1435623000 and unix_time<=1435623600) or # from run 5801
        # (unix_time>=1436394000 and unix_time<=1436395200) or # from run 5845 22 hour balloon launch
        (unix_time>=1437601200 and unix_time<=1437602700) or # from run 5915 looks like error at the start
        # (unix_time>=1439933700 and unix_time<=1439934960) or # from run 6048 22 hour balloon launch
        (unix_time>=1440581700 and unix_time<=1440582480) or # from run 6086
        # (unix_time>=1441489200 and unix_time<=1441490280) or # from run 6137 22 hour balloon launch
        # (unix_time>=1444685400 and unix_time<=1444687080) or # from run 6322 22 hour balloon launch
        # (unix_time>=1445722020 and unix_time<=1445723220) or # from run 6383 22 hour balloon launch
        (unix_time>=1445934900 and unix_time<=1445935500) or # from run 6396
        (unix_time>=1445960400 and unix_time<=1445961000) or # from run 6397
        # (unix_time>=1445982120 and unix_time<=1445982900) or # from run 6398 22 hour balloon launch
        (unix_time>=1446165600 and unix_time<=1446166200) or # from run 6408
        # (unix_time>=1446327300 and unix_time<=1446328200) or # from run 6418 22 hour balloon launch
        (unix_time>=1446607800 and unix_time<=1446608640) or # from run 6433 looks like an error at end
        (unix_time>=1446784200 and unix_time<=1446784800) or # from run 6445
        # (unix_time>=1476739800 and unix_time<=1476741000) or # from run 8100 22 hour balloon launch
        # (unix_time>=1476999000 and unix_time<=1476999900) or # from run 8114 22 hour balloon launch but barely noticeable
        # (unix_time>=1477258200 and unix_time<=1477259100) or # from run 8129 22 hour balloon launch
        (unix_time>=1477511700 and unix_time<=1477512600) or # from run 8143 weird possible balloon launch
        (unix_time>=1477950300 and unix_time<=1477951500) or # from run 8168 22 hour balloon launch
        # (unix_time>=1478033400 and unix_time<=1478034000) or # from run 8173 22 hour balloon launch
        # (unix_time>=1478295300 and unix_time<=1478296200) or # from run 8188 22 hour balloon launch
        # (unix_time>=1478728500 and unix_time<=1478729400) or # from run 8213 22 hour balloon launch
        (unix_time>=1479231900 and unix_time<=1479232500) or # from run 8241

        # config 5
        (unix_time>=1449280500 and unix_time<=1449281100) or # from run 6513
        (unix_time>=1449610200 and unix_time<=1449612000) or # from run 6531
        (unix_time>=1450536000 and unix_time<=1450537200) or # from run 6584
        # (unix_time>=1450906200 and unix_time<=1450907100) or # from run 6606    22hr
        # (unix_time>=1451423700 and unix_time<=1451424600) or # from run 6635   22hr
        (unix_time>=1452008100 and unix_time<=1452009000) or # from run 6669
        # (unix_time>=1452115800 and unix_time<=1452116700) or # from run 6675    22hr
        (unix_time>=1452197700 and unix_time<=1452198600) or # from run 6679
        (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6680
        (unix_time>=1452282000 and unix_time<=1452282600) or # from run 6684
        (unix_time>=1452298200 and unix_time<=1452298800) or # from run 6685    possible error
        (unix_time>=1452385500 and unix_time<=1452386400) or # from run 6690
        # (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6694   22 hr
        (unix_time>=1452494100 and unix_time<=1452495000) or # from run 6696   possible error
        # (unix_time>=1452545100 and unix_time<=1452545880) or # from run 6700    could be error or 22hr
        # (unix_time>=1452636900 and unix_time<=1452637500) or # from run 6705   could be error or 22hr
        (unix_time>=1452715200 and unix_time<=1452716100) or # from run 6709   possible error
        (unix_time>=1452972300 and unix_time<=1452973440) or # from run 6724   possible error
        # (unix_time>=1453325400 and unix_time<=1453326600) or # from run 6743   22 hr
        (unix_time>=1453408500 and unix_time<=1453409400) or # from run 6747
        (unix_time>=1453930200 and unix_time<=1453931400) or # from run 6776
        # (unix_time>=1454535000 and unix_time<=1454536500) or # from run 6818   22 hr
        # (unix_time>=1455746400 and unix_time<=1455747900) or # from run 6889   22 hr
        (unix_time>=1456200900 and unix_time<=1456201800) or # from run 6916
        (unix_time>=1456392600 and unix_time<=1456393800) or # from run 6927
        (unix_time>=1456997400 and unix_time<=1456999200) or # from run 6962
        # (unix_time>=1457559000 and unix_time<=1457560800) or # from run 6994   22 hr
        (unix_time>=1460842800 and unix_time<=1460844600) or # from run 7119   22 hr // has CW contam cal pulsers
        # (unix_time>=1461620100 and unix_time<=1461621900) or # from run 7161   22 hr
        (unix_time>=1463002200 and unix_time<=1463004000) or # from run 7243  22 hr // has CW contam cal pulsers
        (unix_time>=1466501400 and unix_time<=1466503200) or # from run 7474
        (unix_time>=1466721900 and unix_time<=1466724600) or # from run 7486 22 hr // has CW contam cal pulsers
        (unix_time>=1466805600 and unix_time<=1466808300) or # from run 7489 22 hr // has CW contam cal pulsers
        (unix_time>=1466890200 and unix_time<=1466892000) or # from run 7494   22 hr // has CW contam cal pulsers
        (unix_time>=1467927600 and unix_time<=1467929700) or # from run 7552   22 hr
        # (unix_time>=1472333400 and unix_time<=1472335200) or # from run 7831   22 hr
        (unix_time>=1473111300 and unix_time<=1473112800) or # from run 7879    22 hr // has CW contam cal
        # (unix_time>=1473370500 and unix_time<=1473372900) or # from run 7899   22 hr
        # (unix_time>=1475011500 and unix_time<=1475013600) or # from run 7993   22 hr
        (unix_time>=1475185200 and unix_time<=1475187900) or # from run 8003 balloon 22hr // has CW contam cal pulsers
        # (unix_time>=1475358000 and unix_time<=1475359800) or # from run 8013 balloon 22h
        (unix_time>=1475529900 and unix_time<=1475531400) or # from run 8023 balloon 22hr // has CW contam cal pulsers
        # (unix_time>=1475702700 and unix_time<=1475704200) or # from run 8033 balloon 22hr
        (unix_time>=1476221400 and unix_time<=1476222300)): # from run 8069 balloon 22hr // has CW contam cal pulsers
        # (unix_time>=1476479700 and unix_time<=1476481800) # from run 8084 balloon 22hr

            bad_unit_t = True

    elif st == 3:

        # config 1 from undergrads
        if((unix_time>=1380234300 and unix_time<=1380235500) or # from run 1538, 22 hour balloon launch
        (unix_time>=1381008600 and unix_time<=1381010400) or # from run 1584, 22 hour balloon launch
        (unix_time>=1382476200 and unix_time<=1382477400) or # from run 1670, 22 hour balloon launch-ish
        (unix_time>=1382687400 and unix_time<=1382688600) or # from run 1682
        (unix_time>=1382712600 and unix_time<=1382713800) or # from run 1684, 15 hour spike
        (unix_time>=1382972700 and unix_time<=1382973300) or # from run 1698, 15 hour spike
        (unix_time>=1383688800 and unix_time<=1383691500) or # from run 1739, 22 hour balloon launch
        (unix_time>=1384060200 and unix_time<=1384060800) or # from run 1761
        (unix_time>=1384208700 and unix_time<=1384209900) or # from run 1770, 22 hour balloon launch
        (unix_time>=1384486200 and unix_time<=1384492800) or # from run 1786, repeated bursts over ~2 hrs
        (unix_time>=1389399600 and unix_time<=1389400800) or # from run 1980
        (unix_time>=1389744000 and unix_time<=1389747600) or # from run 2001, lots of activity, sweeps in phi
        (unix_time>=1390176600 and unix_time<=1390182000) or # from run 2025
        (unix_time>=1391027700 and unix_time<=1391028900) or # from run 2079, 22 hour balloon launch, but early?
        (unix_time>=1393652400 and unix_time<=1393660800) or # from run 2235, repeated bursts over ~2 hrs
        (unix_time>=1394846400 and unix_time<=1394856000) or # from run 2328, repeated bursts over ~2.5 hours
        (unix_time>=1395437400 and unix_time<=1395438600) or # from run 2363, 22 hour balloon launch
        (unix_time>=1397856300 and unix_time<=1397857800) or # from run 2526, 22 hour balloon launch

        # config 2
        (unix_time>=1390176600 and unix_time<=1390182000) or # from run 3533

        # config 3
        (unix_time>=1409954100 and unix_time<=1409956200) or # from run 3216, 22 hour balloon launch
        (unix_time>=1409986800 and unix_time<=1409988600) or # from run 3217
        (unix_time>=1412026200 and unix_time<=1412028000) or # from run 3332
        (unix_time>=1412284920 and unix_time<=1412287020) or # from run 3347, 22 hour balloon launch
        (unix_time>=1412544120 and unix_time<=1412546400) or # from run 3362, 22 hour balloon launch
        (unix_time>=1412803620 and unix_time<=1412805780) or # from run 3377, 22 hour balloon launch
        (unix_time>=1413897900 and unix_time<=1413899100) or # from run 3439
        (unix_time>=1413914400 and unix_time<=1413922200) or # from run 3440 big wide weird above ground
        (unix_time>=1414083600 and unix_time<=1414086300) or # from run 3449 , 2 spikes
        (unix_time>=1413550800 and unix_time<=1413552600) or # from run 3419, end of the run, before a software dominated run starts
        (unix_time>=1414674000 and unix_time<=1414675500) or # from run 3478
        (unix_time>=1415380500 and unix_time<=1415381400) or # from run 3520
        (unix_time>=1415460600 and unix_time<=1415461500) or # from run 3524
        (unix_time>=1415742000 and unix_time<=1415744100) or # from run 3540 22hr balloon
        (unix_time>=1416207300 and unix_time<=1416209700) or # from run 3568 2 small spikes
        (unix_time>=1416457800 and unix_time<=1416459000) or # from run 3579
        (unix_time>=1416909600 and unix_time<=1416910680) or # from run 3605
        (unix_time>=1416951000 and unix_time<=1416952500) or # from run 3608 22hr balloon
        (unix_time>=1417676400 and unix_time<=1417679400) or # from run 3647
        (unix_time>=1417742400 and unix_time<=1417743600) or # from run 3651
        (unix_time>=1417836600 and unix_time<=1417839300) or # from run 3656
        (unix_time>=1420317000 and unix_time<=1420318200) or # from run 3800
        (unix_time>=1420493700 and unix_time<=1420494600) or # from run 3810 22hr balloon
        (unix_time>=1420513200 and unix_time<=1420515000) or # from run 3811
        (unix_time>=1420598700 and unix_time<=1420600500) or # from run 3816
        (unix_time>=1420857900 and unix_time<=1420859700) or # from run 3830
        (unix_time>=1421019000 and unix_time<=1421020200) or # from run 3840 22hr balloon maybe?
        (unix_time>=1421101800 and unix_time<=1421103600) or # from run 3863 22hr balloon
        (unix_time>=1421723400 and unix_time<=1421723940) or # from run 3910
        (unix_time>=1421750700 and unix_time<=1421751720) or # from run 3912
        (unix_time>=1421868600 and unix_time<=1421881200) or # from run 3977 looks intentional
        (unix_time>=1421881200 and unix_time<=1421884680) or # from run 3978 continuation of thing above
        (unix_time>=1422048900 and unix_time<=1422049800) or # from run 3987 , 22 hour balloon launch
        (unix_time>=1422307200 and unix_time<=1422308100) or # from run 3995 22hr balloon
        (unix_time>=1423660800 and unix_time<=1423661700) or # from run 4132
        (unix_time>=1424819880 and unix_time<=1424820720) or # from run 4200
        (unix_time>=1428529500 and unix_time<=1428531000) or # from run 4412, 22 hour balloon launch
        (unix_time>=1429094400 and unix_time<=1429095600) or # from run 4445
        (unix_time>=1429615800 and unix_time<=1429617600) or # from run 4473
        (unix_time>=1429616700 and unix_time<=1429627500) or # from run 4474
        (unix_time>=1429733400 and unix_time<=1429734600) or # from run 4482
        (unix_time>=1431034500 and unix_time<=1431036900) or # from run 4557 , 22 hour balloon launch
        (unix_time>=1433365500 and unix_time<=1433367900) or # from run 4693
        (unix_time>=1435755600 and unix_time<=1435756500) or # from run 4829
        (unix_time>=1435791000 and unix_time<=1435791600) or # from run 4832
        (unix_time>=1436393700 and unix_time<=1436395500) or # from run 4867
        (unix_time>=1476740100 and unix_time<=1476741300) or # from run 7658
        (unix_time>=1477511400 and unix_time<=1477518300) or # from run 7704, big spike followed by nothing at all
        (unix_time>=1477604700 and unix_time<=1477605900) or # from run 7709,  22 hour balloon launch
        (unix_time>=1477950300 and unix_time<=1477951500) or # from run 7729
        (unix_time>=1479231600 and unix_time<=1479235800) or # from run 7802  , big spike followed by nothing at all

        # config 4
        (unix_time>=1448959200 and unix_time<=1448960100) or # from run 6009
        (unix_time>=1449610500 and unix_time<=1449611400) or # from run 6046 22 hour balloon launch
        (unix_time>=1450119900 and unix_time<=1450120500) or # from run 6077 possible 22 hour balloon launch
        (unix_time>=1450536360 and unix_time<=1450536720) or # from run 6098 spike is at end of time
        (unix_time>=1452116100 and unix_time<=1452116700) or # from run 6188 end of time and possible balloon launch
        (unix_time>=1452196800 and unix_time<=1452198600) or # from run 6193 could be balloon
        (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6194
        (unix_time>=1452282300 and unix_time<=1452282900) or # from run 6198 could be balloon
        (unix_time>=1452298500 and unix_time<=1452299100) or # from run 6199 spike is at end of measured time
        (unix_time>=1452385800 and unix_time<=1452386400) or # from run 6203 spike is at end of measured time
        (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6206 spike is at end of measured time, could be balloon
        (unix_time>=1452494100 and unix_time<=1452494700) or # from run 6208 spike is at end of measured time
        (unix_time>=1452544980 and unix_time<=1452545580) or # from run 6212 could be balloon
        (unix_time>=1452561120 and unix_time<=1452561480) or # from run 6213 spike is at end of measured time
        (unix_time>=1452637020 and unix_time<=1452637260) or # from run 6219 spike is at end of measured time, could be balloon
        (unix_time>=1452715320 and unix_time<=1452715680) or # from run 6223 spike is at end of measured time
        (unix_time>=1452972660 and unix_time<=1452973020) or # from run 6239 spike is at end of measured time
        (unix_time>=1453325400 and unix_time<=1453326300) or # from run 6259 could be balloon
        (unix_time>=1453930500 and unix_time<=1453931100) or # from run 6295 could be balloon
        (unix_time>=1454535000 and unix_time<=1454536200) or # from run 6328 could be balloon
        (unix_time>=1454911200 and unix_time<=1454911800) or # from run 6349 spike is at end of measured time could match below
        (unix_time>=1454911200 and unix_time<=1454912100) or # from run 6350 spike is at start of measured time could match above
        (unix_time>=1455746400 and unix_time<=1455747300) or # from run 6397 could be balloon
        (unix_time>=1456374300 and unix_time<=1456374900) or # from run 6433
        (unix_time>=1457559300 and unix_time<=1457560500) or # from run 6501 could be balloon
        (unix_time>=1460843100 and unix_time<=1460844600) or # from run 6618 spike is at start of measured time, could be balloon
        (unix_time>=1467927840 and unix_time<=1467929640) or # from run 7052 could be balloon
        (unix_time>=1473371280 and unix_time<=1473372180) or # from run 7458 could be balloon
        (unix_time>=1475186100 and unix_time<=1475187000) or # from run 7562 could be balloon
        (unix_time>=1475530500 and unix_time<=1475531700) or # from run 7584 could be balloon
        (unix_time>=1476221400 and unix_time<=1476222600)): # from run 7625 could be balloon        

            bad_unit_t = True

    elif st == 5:
            pass

    return bad_unit_t











