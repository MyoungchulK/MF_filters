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

def get_dag_statement(st, config, run, setup, f_type = None):

    statements = ""
    if f_type is not None:
        statements += f'JOB job_ARA_S{st}_C{config}_R{run}_{f_type} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{st}_C{config}_R{run}_{f_type} setup="{setup}" st="{st}" config="{config}" run="{run}" flavor="{f_type}"\n\n'
    else:
        statements += f'JOB job_ARA_S{st}_C{config}_R{run} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{st}_C{config}_R{run} setup="{setup}" st="{st}" config="{config}" run="{run}"\n\n'

    return statements

setup_path = str(sys.argv[1])
sim_type = str(sys.argv[2])
num_runs = 100

result_path = f'/misc/disk19/users/mkim/OMF_filter/ARA02/sim_{sim_type}'
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(result_path)
result_path = f'/misc/disk19/users/mkim/OMF_filter/ARA03/sim_{sim_type}'
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(result_path)

setup_list = glob(f'{setup_path}*')
print(setup_list)

print('Dag making starts!')
dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{sim_type}/'
if not os.path.exists(dag_path):
    os.makedirs(dag_path)
print(dag_path)
dag_file_name = f'{dag_path}A.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

    for s in setup_list:
        st = int(get_path_info(s, '/A', '_C'))        
        config = int(get_path_info(s, '_C', '_E'))
        if sim_type == 'signal':
            f_type = str(get_path_info(s, '_E100_Nu', '_signal'))
        else:
            f_type = None        

        for r in range(num_runs):
            statements = get_dag_statement(st, config, r, s, f_type)
            with open(dag_file_name, 'a') as f:
                f.write(statements)
        
print('Dag making is done!')
 
