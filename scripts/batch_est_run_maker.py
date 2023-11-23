import numpy as np
import os, sys
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def get_dag_statement(st, pol, slo, frac):

        statements = ""
        statements += f'JOB job_ARA_S{st}_P{pol}_S{slo}_F{frac} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{st}_P{pol}_S{slo}_F{frac} station="{st}" pol="{pol}" slo="{slo}" frac="{frac}"\n\n'

        return statements

def batch_run_loader(Station = None, Output = None):

    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)


    print('Dag making starts!')
    dag_file_name = f'{Output}A{Station}.dag'
    statements = ""

    with open(dag_file_name, 'w') as f:
        f.write(statements)

    for p in range(2):
                for s in tqdm(range(180)):
                    for f in range(20):
                        statements = get_dag_statement(Station, p, s, f)
                        with open(dag_file_name, 'a') as f:
                            f.write(statements)    


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

    batch_run_loader(Station = station, Output = output)
 
