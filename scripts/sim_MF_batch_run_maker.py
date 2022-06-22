import os, sys
import numpy as np
from glob import glob
from tqdm import tqdm

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

def get_dag_statement(d_count, d, st, year, d_type):

    statements = ""
    statements += f'JOB job_ARA_S{st}_{d_type}_{d_count} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_{d_type}_{d_count} data="{d}" st="{st}" year="{year}" d_type="{d_type}" d_count="{d_count}"\n\n'

    return statements

st = int(sys.argv[1])
d_type = str(sys.argv[2])
data_path = str(sys.argv[3])

data_list = glob(f'{data_path}AraOut*')
print(data_list)

print('Dag making starts!')
dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_mf/'
if not os.path.exists(dag_path):
    os.makedirs(dag_path)
print(dag_path)
dag_file_name = f'{dag_path}A{st}_{d_type}.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)
    
    d_count = 0
    for d in tqdm(data_list):
        
        st = int(get_path_info(d, 'AraOut.A', '_C'))
        config = int(get_path_info(d, '_C', '_E'))
        if st == 3 and config > 5:
            year = 2018
            print(d)
        else:   
            year = 2015

        statements = get_dag_statement(d_count, d, st, year, d_type)
        with open(dag_file_name, 'a') as f:
            f.write(statements)
        d_count += 1        

print('Dag making is done!')












 
