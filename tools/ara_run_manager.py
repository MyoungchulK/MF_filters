import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

class run_info_loader:

    def __init__(self, st, run):

        self.st = st
        self.run = run

    def get_ped_path(self, verbose = False, return_none = False):

        ped_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/Ped/pedestalValues.run{self.run}.dat'

        if os.path.exists(ped_path):
            if verbose:
                print(f'ped_path:{ped_path}')
        else:
            print('There is no desired pedestal!')
            if return_none == True:
                return None
            else:
                sys.exit(1)

        return ped_path

    def get_6_digit_run_number(self):

        run_6_digit = list('000000')
        run_6_digit[-len(str(self.run)):] = str(self.run)
        run_6_digit = "".join(run_6_digit)

        return run_6_digit

    def get_data_path(self, file_type = 'event', analyze_blind_dat = False, verbose = False, return_none = False):

        if analyze_blind_dat == True:
            a2_2013_run_limit = 2820
            a3_2013_run_limit = 1930
            dat_type_2013 = 'full2013Data'
            dat_type = 'blinded'
        else:
            a2_2013_run_limit = 2793
            a3_2013_run_limit = 1902
            dat_type_2013 = 'burnSample1in10'
            dat_type = 'unblinded'
        
        if (self.st == 2 and self.run < a2_2013_run_limit) or (self.st == 3 and self.run < a3_2013_run_limit):
            dat_ls_path = f'/data/exp/ARA/2013/filtered/{dat_type_2013}/ARA0{self.st}/root/run{self.run}/'
            dat_name = f'{file_type}{self.run}.root'
        else:
            run_6_digit = self.get_6_digit_run_number()
            dat_ls_path = f'/data/exp/ARA/*/{dat_type}/L1/ARA0{self.st}/*/run{run_6_digit}/'
            dat_name = f'{file_type}{run_6_digit}.root'
        dat_goal = dat_ls_path + dat_name
        dat_path = glob(dat_goal)
        dat_ls_path = glob(dat_ls_path)

        if len(dat_path) != 1:
            if self.st == 3 and (self.run == 2198 or self.run == 3517):
                dat_path = dat_path[0]
            else:
                print('There is no desired data!')
                print(f'File on the search: {dat_goal}')
                print(f'Possible location: {dat_ls_path}')
                print('Files in the location:', os.listdir(dat_ls_path[0]))
                if return_none == True:
                    return None
                else:
                    sys.exit(1)
        else:
            dat_path = dat_path[0]
      
        if os.path.exists(dat_path):
            if verbose:
                print(f'dat_path:{dat_path}')
        else:
            print('There is no desired data!')
            if return_none == True:
                return None
            else:
                sys.exit(1)

        return dat_path

    def get_data_ped_path(self, file_type = 'event', analyze_blind_dat = False, verbose = False, return_none = False):

        dat_path = self.get_data_path(file_type = file_type, analyze_blind_dat = analyze_blind_dat, verbose = verbose, return_none = return_none)
        ped_path = self.get_ped_path(verbose = verbose, return_none = return_none)

        return dat_path, ped_path

    def get_data_info(self):

        # salvage just number
        dat_path = self.get_data_path()
        dat_path_num = re.sub("\D", "", dat_path)

        config = self.get_config_number() 
        year = int(dat_path_num[:4])
        if year == 2013:
            station = int(dat_path_num[7:9])
            run_num = int(re.sub("\D", "", dat_path[-11:]))
            month = -1
            date = -1
        else:
            station = int(dat_path_num[5:7])
            run_num = int(dat_path_num[11:17])
            month = int(dat_path_num[7:9])
            date = int(dat_path_num[9:11])

        if self.st != station or self.run != run_num:
            print('Station and Run number are different!')
            sys.exit(1)
        else:
            print(f'data info. 1)station:{station} 2)run:{run_num} 3)config:{config} 4)year:{year} 5)month:{month} 6)date:{date}')

        return station, run_num, config, year, month, date

    def get_config_number(self):

        # from Brian: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_Cuts.h

        # default. unknown
        config=0

        # by statation
        if self.st == 2:
            if self.run>=0 and self.run<=4:
                config=1
            elif self.run>=11 and self.run<=60:
                config=4
            elif self.run>=120 and self.run<=2274:
                config=2
            elif self.run>=2275 and self.run<=3463:
                config=1
            elif self.run>=3465 and self.run<=4027:
                config=3
            elif self.run>=4029 and self.run<=6481:
                config=4
            elif self.run>=6500 and self.run<=8097:
                config=5
            #elif self.run>=8100 and self.run<=8246:
            elif self.run>=8100 and self.run<=9504:
                config=4
            elif self.run>=9505 and self.run<=9748:
                config=5
            elif self.run>=9749:
                config=6
            else:
                pass

        elif self.st == 3:
            if self.run>=0 and self.run<=4:
                config=1
            elif self.run>=470 and self.run<=1448:
                config=2
            elif self.run>=1449 and self.run<=1901:
                config=1
            elif self.run>=1902 and self.run<=3103:
                config=5
            elif self.run>=3104 and self.run<=6004:
                config=3
            elif self.run>=6005 and self.run<=7653:
                config=4
            elif self.run>=7658 and self.run<=7808:
                config=3
            elif self.run>=10001:
                config=6
            else:
                pass

        elif self.st == 5:
            pass

        return config

class batch_info_loader:

    def __init__(self, st):

        self.st = st
        self.years = np.arange(2013,2019, dtype = int)
        if self.st == 3:
            self.years = self.years[self.years != 2017]

    def get_dag_statement(self, run):

        statements = ""
        statements += f'JOB job_ARA_S{self.st}_R{run} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{self.st}_R{run} station="{self.st}" run="{run}"\n\n'

        return statements

    def get_dag_file(self, path, file_type = 'event', analyze_blind_dat = False):

        lists = self.get_dat_list(file_type = file_type, analyze_blind_dat = analyze_blind_dat)

        self.get_list_in_txt(path, lists)

        print('Dag making starts!')
        dag_file_name = f'{path}A{self.st}.dag'
        statements = ""

        with open(dag_file_name, 'w') as f:
            f.write(statements)

            for w in tqdm(lists[0]):
                statements = self.get_dag_statement(int(w))
                with open(dag_file_name, 'a') as f:
                    f.write(statements)

        print('Dag making is done!')
        print(f'output is {dag_file_name}')

    def get_list_in_txt(self, path, lists):

        print('Text list making starts!')
        txt_path = f'{path}/txt/'
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        label = ['run', 'bad_run', 'dupl_run']
        for t in range(3):
            print(f'Make {label[t]} list')
            
            txt_name = f'{txt_path}A{self.st}_{label[t]}_list.txt'
            with open(txt_name, 'w') as f:
                for l in tqdm(range(len(lists[t*2]))):
                    statements = f'{lists[2*t][l]} {lists[2*t + 1][l]} \n'
                    f.write(statements)                    

        print('Text list making is done!')

    def get_dat_list(self, file_type = 'event', analyze_blind_dat = False):
        
        if analyze_blind_dat == True:
            dat_type_2013 = 'full2013Data'
            dat_type = 'blinded'
        else:
            dat_type_2013 = 'burnSample1in10'
            dat_type = 'unblinded'
        
        run_list = []
        dat_list = []
        bad_run_list = []
        bad_dat_list =[]
        for yrs in self.years:
            if int(yrs) == 2013:
                dat_path = f'/data/exp/ARA/{int(yrs)}/filtered/{dat_type_2013}/ARA0{self.st}/root/run[0-9]*/{file_type}[0-9]*.root' 
            else:
                dat_path = f'/data/exp/ARA/{int(yrs)}/{dat_type}/L1/ARA0{self.st}/[0-9][0-9][0-9][0-9]/run[0-9][0-9][0-9][0-9][0-9][0-9]/{file_type}[0-9][0-9][0-9][0-9][0-9][0-9].root'

            print(dat_path) 
            run_yrs_list = []
            dat_yrs_list_old = glob(dat_path)
            dat_yrs_list = []
            for d in dat_yrs_list_old:
                num_in_str = d[-11:]
                if num_in_str.find('_') != -1:
                    continue
                run_num = int(re.sub("\D", "", num_in_str))
                run_yrs_list.append(run_num)
                dat_yrs_list.append(d)
                del run_num
            run_yrs_list = np.asarray(run_yrs_list)

            run_yrs_idx = np.argsort(run_yrs_list)        
            run_yrs_sort = run_yrs_list[run_yrs_idx]            
            dat_yrs_sort = []
            for s in run_yrs_idx:
                dat_yrs_sort.append(dat_yrs_list[int(s)])

            if self.st == 3 and int(yrs) == 2018:
                wrong_idx = run_yrs_sort < 10000
            else:
                wrong_idx = run_yrs_sort < 100
            bad_run = run_yrs_sort[wrong_idx]
            bad_dat = []
            print(f'{int(yrs)} Wrong Run#!:{bad_run}')
            for w in range(len(wrong_idx)):
                if wrong_idx[w]:
                    bad_dat.append(dat_yrs_sort[w])
                    print(dat_yrs_sort[w])
            bad_run_list.extend(bad_run)
            bad_dat_list.extend(bad_dat)

            right_idx = ~wrong_idx
            run_yrs_sort_right = run_yrs_sort[right_idx]
            dat_yrs_sort_right = []
            for r in range(len(dat_yrs_sort)):
                if right_idx[r]:
                    dat_yrs_sort_right.append(dat_yrs_sort[r])

            run_list.extend(run_yrs_sort_right)
            dat_list.extend(dat_yrs_sort_right)
        run_list = np.asarray(run_list)
        bad_run_list = np.asarray(bad_run_list)

        # duplication check
        dupl_run_list = []
        dupl_dat_list = []
        dupl = np.unique(run_list, return_index = True, return_counts=True)
        dupl_run_num = dupl[0][dupl[2] > 1]
        for d in dupl_run_num:
            d = int(d)
            dupl_run_idx = np.where(run_list == d)[0]
            print(f'Duplicated Runs:{d}')
            for i in dupl_run_idx:    
                i = int(i)
                dupl_path = dat_list[i]
                print(f'{dupl_path}')
                dupl_run_list.append(d)
                dupl_dat_list.append(dupl_path)
        dupl_run_list = np.asarray(dupl_run_list)

        run_list = run_list[dupl[1]]
        dat_list_new = []
        for d in dupl[1]:
            d = int(d)
            dat_list_new.append(dat_list[d])
        print(f'Total number of runs is {len(run_list)}')
       
        # final check
        for f in range(len(run_list)):
            run_num = int(re.sub("\D", "", dat_list_new[f][-11:]))
            if run_list[f] != run_num:
                print(run_list[f], run_num)
                print('Run number doesnt match!')
                sys.exit(1)
        print('All Data has a perfect match!')

        return run_list, dat_list_new, bad_run_list, bad_dat_list, dupl_run_list, dupl_dat_list

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

