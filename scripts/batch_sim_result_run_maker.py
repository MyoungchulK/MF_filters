import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

def get_dag_statement(d_count, d_name, st):

    statements = ""
    statements += f'JOB job_ARA_S{st}_R{d_count} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_R{d_count} data="{d_name}" st="{st}" run="{d_count}"\n\n'

    return statements

st = int(sys.argv[1])
s_type = str(sys.argv[2])
d_path = str(sys.argv[3])
d_list = glob(f'{d_path}*')

print('Dag making starts!')
dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{s_type}_result/'
if not os.path.exists(dag_path):
    os.makedirs(dag_path)
print(dag_path)

dag_file_name = f'{dag_path}A{st}_g.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

    d_count = 0
    for d in tqdm(d_list):
        
        statements = get_dag_statement(d_count, d, st)
        with open(dag_file_name, 'a') as f:
            f.write(statements)
        d_count += 1        

print('Dag making is done!')
 
