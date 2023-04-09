import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
import uproot

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_run_manager import get_path_info_v2

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)

def get_dag_statement(st, run, sim_run, flavors_int, energy_int):

    statements = ""
    statements += f'JOB job_ARA_E{energy_int}_F{flavors_int}_S{st}_R{run}_Sim{sim_run} ARA_job.sub \n'
    statements += f'VARS job_ARA_E{energy_int}_F{flavors_int}_S{st}_R{run}_Sim{sim_run} en="{energy_int}" fla="{flavors_int}" st="{st}" run="{run}" sim_run="{sim_run}"\n\n'

    return statements

def dag_maker(data):
    config = int(get_path_info_v2(data, '_R', '.txt'))
    flavor = int(get_path_info_v2(data, '_F', '_A'))
    sim_run = int(get_path_info_v2(data, 'txt.run', '.root'))
    energy = int(get_path_info_v2(data, '_E', '_F'))
    print('St:', Station, 'Config:', config, 'SimRun:', sim_run, 'Energy:', energy, 'Flavor:', flavor)

    statements = get_dag_statement(Station, config, sim_run, flavor, energy)
    with open(dag_file_name, 'a') as f:
        f.write(statements)

err_counts = 0
num_counts = 0
tot_counts = 0
dag_file_name = f'/home/mkim/A{Station}.dag'
statements = ""
with open(dag_file_name, 'w') as f:
    f.write(statements)

for r in tqdm(range(len(d_run_tot))):
    
    try: 
        file_uproot = uproot.open(d_list[r])   
    except ValueError:
        print('Error!', d_list[r])
        dag_maker(d_list[r])
        err_counts += 1
        continue
    pnu = np.asarray(file_uproot['AraTree2/event/pnu'], dtype = float)
    if len(pnu) != 100:
        print('Number!', d_list[r])
        dag_maker(d_list[r])
        num_counts += 1
    del file_uproot, pnu

tot_counts = err_counts + num_counts
print('Tot:', tot_counts, 'Err:', err_counts, 'Num:', num_counts)
 
