import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

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

def get_dag_statement(d, root_name, d_count):

    statements = ""
    statements += f'JOB job_ARA_R{d_count} ARA_job.sub \n'
    statements += f'VARS job_ARA_R{d_count} data="{d}" output="{root_name}" run="{d_count}"\n\n'

    return statements


d_path = str(sys.argv[1]) # /misc/disk19/users/mkim/OMF_filter/radiosonde_data/MWX/*/*
r_path = str(sys.argv[2]) # /misc/disk19/users/mkim/OMF_filter/radiosonde_data/root/

if not os.path.exists(r_path):
    os.makedirs(r_path)
print(r_path)

d_list = glob(f'{d_path}*/*/*')
print(d_list)
d_front_mane = 'NZSP_'

print('Dag making starts!')
dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_mwx/'
if not os.path.exists(dag_path):
    os.makedirs(dag_path)
print(dag_path)
shell_file_name = f'{dag_path}A.sh'
statements = ""
with open(shell_file_name, 'w') as f:
    f.write(statements)

    for d in tqdm(d_list):
        file_name = get_path_info(d, d_front_mane, '.mwx')
        root_name = r_path + d_front_mane + file_name + '.root'
        sline = f'/home/mkim/analysis/AraSoft/mwx2root/mwx2root -o {root_name} {d}\n'
        with open(shell_file_name, 'a') as f:
            f.write(sline)

dag_file_name = f'{dag_path}A.dag'
with open(dag_file_name, 'w') as f:
    f.write(statements)

    d_count = 0
    for d in tqdm(d_list):
        
        file_name = get_path_info(d, d_front_mane, '.mwx')
        root_name = r_path + d_front_mane + file_name + '.root' 

        statements = get_dag_statement(d, root_name, d_count)
        with open(dag_file_name, 'a') as f:
            f.write(statements)
        d_count += 1        

print('Dag making is done!')
 
