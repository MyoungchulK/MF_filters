import os, sys
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from datetime import datetime
from subprocess import call

def get_dag_statement(evt_range):

    data = '/data/user/mkim/OMF_filter/ARA02/psd_sim/AraOut.setup_A2_noise_evt10000.txt.run0.root'

    statements = ""
    statements += f'JOB job_ARA_S2_E{evt_range} ARA_job.sub \n'
    statements += f'VARS job_ARA_S2_E{evt_range} data="{data}" evt="{evt_range},{evt_range+10}" station="2" run="{evt_range}"\n\n'

    return statements

path = '/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_noise/' 
if not os.path.exists(path):
    os.makedirs(path)

print('Dag making starts!')
dag_file_name = f'{path}A2.dag'
statements = ""

evt_range = np.arange(0,10000,10,dtype = int)
evt_len = len(evt_range)

with open(dag_file_name, 'w') as f:
    f.write(statements)

    for w in tqdm(range(evt_len)):
        statements = get_dag_statement(evt_range[w])
        with open(dag_file_name, 'a') as f:
            f.write(statements)

print('Dag making is done!')
 
