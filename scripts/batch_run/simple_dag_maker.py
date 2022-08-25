import os, sys
import numpy as np
from glob import glob
from tqdm import tqdm

st = int(sys.argv[1])

dag_path = str(sys.argv[2])
if not os.path.exists(dag_path):
    os.makedirs(dag_path)

dag_file_name = f'{dag_path}A{st}.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

    trig = np.arange(3, dtype = int)
    run_w = 100
    run = np.arange(0, 11000, run_w, dtype = int)

    for t in range(len(trig)):
        for r in range(len(run)):

            statements = ""
            statements += f'JOB job_ARA_S{st}_T{trig[t]}_R{run[r]} ARA_job.sub \n'
            statements += f'VARS job_ARA_S{st}_T{trig[t]}_R{run[r]} station="{st}" trig="{trig[t]}" run="{run[r]}" run_w="{run_w}"\n\n'

            with open(dag_file_name, 'a') as f:
                f.write(statements)
        
print('Dag making is done!')
 
