import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
import uproot

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_run_manager import get_path_info_v2

def get_dag_statement(st, run, sim_run, flavors_int, energy_int):

    statements = ""
    statements += f'JOB job_ARA_E{energy_int}_F{flavors_int}_S{st}_R{run}_Sim{sim_run} ARA_job.sub \n'
    statements += f'VARS job_ARA_E{energy_int}_F{flavors_int}_S{st}_R{run}_Sim{sim_run} en="{energy_int}" fla="{flavors_int}" st="{st}" run="{run}" sim_run="{sim_run}"\n\n'

    return statements

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sim_signal_full/'

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
num_flas = 3
num_sim_runs = 80
energy = np.arange(16, 21, 1, dtype = int)
num_ens = len(energy)

err_counts = 0
num_counts = 0
tot_counts = 0
miss_counts = 0
dag_file_name = f'/home/mkim/A{Station}.dag'
statements = ""
with open(dag_file_name, 'w') as ff:
    ff.write(statements)

pbar = tqdm(total = num_configs * num_flas * num_sim_runs * num_ens)
for c in range(num_configs):
    for f in range(num_flas):
        for r in range(num_sim_runs):
            for e in range(num_ens):
                pbar.update(1)

                con = int(c + 1)
                fla = int(f + 1)
                run = int(r)
                en = energy[e]

                file_name = f'{d_path}AraOut.signal_E{en}_F{fla}_A{Station}_R{con}.txt.run{run}.root'
                if os.path.exists(file_name):
                    #print(file_name)
                    try:
                        file_uproot = uproot.open(file_name)
                    except ValueError: 
                        statements = get_dag_statement(Station, con, run, fla, en)
                        with open(dag_file_name, 'a') as ff:
                            ff.write(statements)
                        err_counts += 1
                        print('Error!:', file_name)          
                        continue
                    try:
                        pnu = np.asarray(file_uproot['AraTree2/event/pnu'], dtype = float)
                    except uproot.exceptions.KeyInFileError:
                        statements = get_dag_statement(Station, con, run, fla, en)
                        with open(dag_file_name, 'a') as ff:
                            ff.write(statements)
                        err_counts += 1
                        print('Error v2!:', file_name)
                        continue
                    if len(pnu) != 100:
                        statements = get_dag_statement(Station, con, run, fla, en)
                        with open(dag_file_name, 'a') as ff:
                            ff.write(statements)                
                        num_counts += 1
                        print('Number!:', file_name) 
                else:
                    statements = get_dag_statement(Station, con, run, fla, en)
                    with open(dag_file_name, 'a') as ff:
                        ff.write(statements)
                    miss_counts += 1
                    print('MISS!:', file_name)
pbar.close()

tot_counts = err_counts + num_counts + miss_counts
print('Tot:', tot_counts, 'Err:', err_counts, 'Num:', num_counts, 'Miss:', miss_counts)
 
