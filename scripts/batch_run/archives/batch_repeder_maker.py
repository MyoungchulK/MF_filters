import numpy as np
import os, sys
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def batch_repeder_loader(Station = None, Output = None):

    batch_info = batch_info_loader(Station)
    lists = batch_info.get_dat_list(analyze_blind_dat = True)

    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    print('Dag making starts!')
    dag_file_name = f'{Output}A{Station}.dag'
    statements = ""

    with open(dag_file_name, 'w') as f:
        f.write(statements)

        d = 0
        for w in tqdm(lists[0]):
            statements = get_dag_statement(lists[1][d], Station, int(w))
            with open(dag_file_name, 'a') as f:
                f.write(statements)
            d+= 1

    print('Dag making is done!')
    print(f'output is {dag_file_name}')

def get_dag_statement(data, st, run):

    statements = ""
    statements += f'JOB job_ARA_S{st}_R{run} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_R{run} data="{data}" station="{st}" run="{run}"\n\n'

    return statements

if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output=str(sys.argv[2])

    batch_repeder_loader(Station = station, Output = output)
 
