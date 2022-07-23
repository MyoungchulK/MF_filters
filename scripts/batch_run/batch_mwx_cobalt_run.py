import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call
from subprocess import run

def get_path_info(dat_path, mask_key, end_key):

        mask_idx = dat_path.find(mask_key)
        if mask_idx == -1:
            print('Cannot scrap the info from path!')
            sys.exit(1)
        mask_len = len(mask_key)
        end_idx = dat_path.find(end_key, mask_idx + mask_len)
        val = dat_path[mask_idx + mask_len:end_idx]
        del mask_idx, mask_len, end_idx

        return val

d_path = str(sys.argv[1]) # /misc/disk19/users/mkim/OMF_filter/radiosonde_data/MWX/
r_path = str(sys.argv[2]) # /misc/disk19/users/mkim/OMF_filter/radiosonde_data/root/

if not os.path.exists(r_path):
    os.makedirs(r_path)
print(r_path)

d_list = glob(f'{d_path}*/*/*')
print(len(d_list))
d_front_mane = 'NZSP_'

print('cobalt run starts!')
dag_file_name = f'{r_path}A.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

    d_count = 0
    for d in d_list:
        
        file_name = get_path_info(d, d_front_mane, '.mwx')
        root_name = r_path + d_front_mane + file_name + '.root' 

        CMD_cd = 'cd /home/mkim/analysis/AraSoft/mwx2root'
        CMD_line = f'./mwx2root -o {root_name} {d}'
        #CMD_line = f'./mwx2root'
        #r_line = f'-o {root_name}'
        #d_line = f'{d}'
        #CMD_line = f'./misc/home/mkim/analysis/AraSoft/mwx2root/mwx2root -o {root_name} {d}'
        #print(CMD_line.split(' '))
        print(CMD_line)
    
        #try:
        run(CMD_cd.split(' '), shell=True)
        #print(os.getcwd())
        #call(CMD_line.split(' '), shell=True) 
        run(CMD_line.split(' '), shell=True) 
        #call(f'{CMD_line} {r_line} {d_line}', shell=True) 
        
        #except:
            #with open(dag_file_name, 'a') as f:
                #f.write(CMD_line + '\n\n')
        d_count += 1        

print('cobalt run is done!')
 
