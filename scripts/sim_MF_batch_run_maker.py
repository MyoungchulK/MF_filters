import os, sys
import numpy as np
from glob import glob
from tqdm import tqdm

def get_dag_statement(d_count, d, st, d_type):

    statements = ""
    statements += f'JOB job_ARA_S{st}_{d_type}_{d_count} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_{d_type}_{d_count} data="{d}" st="{st}" d_type="{d_type}" d_count="{d_count}"\n\n'

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
        statements = get_dag_statement(d_count, d, st, d_type)
        with open(dag_file_name, 'a') as f:
            f.write(statements)
        d_count += 1        

print('Dag making is done!')
 
