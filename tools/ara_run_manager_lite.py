import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

class run_info_loader:

    def __init__(self, st, run, analyze_blind_dat = False):

        self.st = st
        self.run = run
        self.analyze_blind_dat = analyze_blind_dat

    def get_ped_path(self, file_type = 'values', verbose = False, return_none = False):

        ped_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/ped_full/ped_full_{file_type}_A{self.st}_R{self.run}.dat'

        if os.path.exists(ped_path):
            if verbose:
                print(f'ped_path:{ped_path}')
        else:
            print(f'There is no desired {ped_path}')
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

    def get_data_path(self, file_type = 'event', verbose = False, return_none = False):

        if self.analyze_blind_dat == True:
            a2_2013_run_limit = 2820
            a3_2013_run_limit = 1930
            dat_type_2013 = 'full2013Data'
            dat_type = 'blinded'
        else:
            a2_2013_run_limit = 2793
            a3_2013_run_limit = 1902
            dat_type_2013 = 'burnSample1in10'
            dat_type = 'unblinded'
       
        if self.analyze_blind_dat == True and self.st == 2 and self.run == 2814: 
            run_6_digit = self.get_6_digit_run_number()
            dat_ls_path = os.path.expandvars("$RAW_PATH") + f'/*/{dat_type}/L1/ARA0{self.st}/*/run{run_6_digit}/'
            dat_name = f'{file_type}{run_6_digit}.root' 
        elif (self.st == 2 and self.run < a2_2013_run_limit) or (self.st == 3 and self.run < a3_2013_run_limit):
            dat_ls_path = os.path.expandvars("$RAW_PATH") + f'/2013/filtered/{dat_type_2013}/ARA0{self.st}/root/run{self.run}/'
            if self.analyze_blind_dat == True and self.st == 2 and (self.run == 2811 or self.run == 2812 or self.run == 2813 or self.run == 2815 or self.run == 2816 or self.run == 2818 or self.run == 2819):
                run_6_digit = self.get_6_digit_run_number()
                dat_name = f'{file_type}{run_6_digit}.root'
            elif self.analyze_blind_dat == True and self.st == 3 and (self.run == 1922 or self.run == 1924 or self.run == 1925 or self.run == 1926 or self.run == 1929):
                run_6_digit = self.get_6_digit_run_number()
                dat_name = f'{file_type}{run_6_digit}.root'
            elif self.analyze_blind_dat == False and self.st == 3 and self.run == 1746:
                run_6_digit = self.get_6_digit_run_number()
                dat_name = f'{file_type}{run_6_digit}.root'
            else:
                dat_name = f'{file_type}{self.run}.root'
        else:
            run_6_digit = self.get_6_digit_run_number()
            dat_ls_path = os.path.expandvars("$RAW_PATH") + f'/*/{dat_type}/L1/ARA0{self.st}/*/run{run_6_digit}/'
            dat_name = f'{file_type}{run_6_digit}.root'
        dat_goal = dat_ls_path + dat_name
        dat_path = glob(dat_goal)
        dat_ls_path = glob(dat_ls_path)

        if len(dat_path) != 1:
            if self.analyze_blind_dat == False and self.st == 3 and (self.run == 2198 or self.run == 3517):
                dat_path = dat_path[0]
            else:
                print(f'There is no desired {file_type} data!')
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
                print(f'{file_type}_dat_path:{dat_path}')
        else:
            print(f'There is no desired {file_type} data!')
            if return_none == True:
                return None
            else:
                sys.exit(1)

        return dat_path

    def get_data_ped_path(self, file_type = 'event', verbose = False, return_none = False, return_dat_only = False):

        dat_path = self.get_data_path(file_type = file_type, verbose = verbose, return_none = return_none)
        if return_dat_only == True:
            ped_path = '0'
        else:
            ped_path = self.get_ped_path(verbose = verbose, return_none = return_none)

        return dat_path, ped_path

    def get_path_info(self, dat_path, mask_key, end_key):

        mask_idx = dat_path.find(mask_key)
        if mask_idx == -1:
            print('Cannot scrap the info from path!')
            sys.exit(1)
        mask_len = len(mask_key)
        end_idx = dat_path.find(end_key, mask_idx + mask_len)        
        val = dat_path[mask_idx + mask_len:end_idx]
        del mask_idx, mask_len, end_idx

        return val

    def get_data_info(self):

        # salvage just number
        dat_path = self.get_data_path()

        config = self.get_config_number() 
        year = int(self.get_path_info(dat_path, 'ARA/', '/'))
        if year == 2013:
            if self.analyze_blind_dat == True:
                station = int(self.get_path_info(dat_path, 'full2013Data/ARA', '/'))
            else:
                station = int(self.get_path_info(dat_path, 'burnSample1in10/ARA', '/'))
            run_num = int(self.get_path_info(dat_path, 'run', '/'))
            month = -1
            date = -1
        else:
            station = int(self.get_path_info(dat_path, 'L1/ARA', '/'))
            run_num = int(self.get_path_info(dat_path, 'run', '/'))
            mmdd = self.get_path_info(dat_path, f'ARA0{station}/', '/')
            month = int(mmdd[:2])
            date = int(mmdd[2:])
            del mmdd

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
