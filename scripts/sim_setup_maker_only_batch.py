import os, sys
import numpy as np
from tqdm import tqdm
import click

def get_dag_statement(st, run, sim_run, flavors = None):

    statements = ""
    if flavors is not None:
        flavors_int = int(flavors)
        statements += f'JOB job_ARA_F{flavors_int}_S{st}_R{run}_Sim{sim_run} ARA_job.sub \n'
        statements += f'VARS job_ARA_F{flavors_int}_S{st}_R{run}_Sim{sim_run} fla="{flavors_int}" st="{st}" run="{run}" sim_run="{sim_run}"\n\n'
    else:
        statements += f'JOB job_ARA_S{st}_R{run}_Sim{sim_run} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{st}_R{run}_Sim{sim_run} st="{st}" run="{run}" sim_run="{sim_run}"\n\n'

    return statements

@click.command()
@click.option('-k', '--key', type = str)
@click.option('-s', '--station', type = int)
@click.option('-b', '--blind_dat', default = False, type = bool)
def main(key, station, blind_dat):
    
    blind = ''
    if blind_dat:
        blind = '_full'
    dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{key}_config/' # dag path
    if not os.path.exists(dag_path):
        os.makedirs(dag_path)
    dag_file_name = f'{dag_path}A{station}.dag'
    statements = ""
    with open(dag_file_name, 'w') as f:
        f.write(statements)

    if station == 2:
        num_configs = 7
    if station == 3:
        num_configs = 9
    if key == 'signal':
        num_runs = 3000
    else:
        num_runs = 1000

    for r in tqdm(range(num_configs)):
        if key == 'signal':
            for f in range(3):
                for s in tqdm(range(num_runs)):
                    fla_idx = int(f + 1)
                    statements = get_dag_statement(station, int(r + 1), s, flavors = fla_idx)
                    with open(dag_file_name, 'a') as ff:
                        ff.write(statements)
                    del statements

        if key == 'noise':
            for s in tqdm(range(num_runs)):
                statements = get_dag_statement(station, int(r + 1), s)
                with open(dag_file_name, 'a') as ff:
                    ff.write(statements)

    print('Done!')

if __name__ == "__main__":

    main()

 
