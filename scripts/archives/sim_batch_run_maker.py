import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

def get_dag_statement(data):

    run = int(data[-11:-5])

    statements = ""
    statements += f'JOB job_ARA_S2_R{run} ARA_job.sub \n'
    statements += f'VARS job_ARA_S2_R{run} data="{data}" run="{run}"\n\n'

    return statements

lists = glob('/data/user/brianclark/for_Uzair/E2_nomag_art/*')
print(len(lists))

path = '/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim/' 
if not os.path.exists(path):
    os.makedirs(path)

print('Dag making starts!')
dag_file_name = f'{path}A2.dag'
statements = ""

with open(dag_file_name, 'w') as f:
    f.write(statements)

    for w in tqdm(lists):
        statements = get_dag_statement(w)
        with open(dag_file_name, 'a') as f:
            f.write(statements)

print('Dag making is done!')
 
