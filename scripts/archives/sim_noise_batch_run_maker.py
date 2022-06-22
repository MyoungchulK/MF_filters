import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

def get_dag_statement(st, config, run, setup, result):

    statements = ""
    statements += f'JOB job_ARA_S{st}_C{config}_R{run} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_C{config}_R{run} setup="{setup}" result="{result}" st="{st}" config="{config}" run="{run}"\n\n'

    return statements


st = int(sys.argv[1])
if st == 2:
    num_configs = 6
if st == 3:
    num_configs = 7 
num_runs = 10

result_path = f'/misc/disk19/users/mkim/OMF_filter/ARA0{st}/sim_noise'
if not os.path.exists(result_path):
    os.makedirs(result_path)
setup_path = '/home/mkim/analysis/MF_filters/sim/sim_noise/'

print('Dag making starts!')
dag_path = '/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_noise/'
if not os.path.exists(dag_path):
    os.makedirs(dag_path)
dag_file_name = f'{dag_path}A{st}.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

    for c in range(num_configs):
        config = c+1
        setup_name = f'A{st}_C{config}_E10000_noise_rayl.txt'
        setup_com = setup_path + setup_name

        for r in range(num_runs):
            statements = get_dag_statement(st, config, r, setup_com, result_path)
            with open(dag_file_name, 'a') as f:
                f.write(statements)

print('Dag making is done!')
 
