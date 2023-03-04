import os, sys
import numpy as np
from tqdm import tqdm
import click

def get_dag_statement(st, en, sim_run):

    statements = ""
    statements += f'JOB job_ARA_S{st}_E{en}_Sim{sim_run} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_E{en}_Sim{sim_run} st="{st}" energy="{en}" sim_run="{sim_run}"\n\n'

    return statements

@click.command()
@click.option('-s', '--station', type = int)
def main(station):
    
    dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_signal_mono/' # dag path
    if not os.path.exists(dag_path):
        os.makedirs(dag_path)
    dag_file_name = f'{dag_path}A{station}.dag'
    statements = ""
    with open(dag_file_name, 'w') as f:
        f.write(statements)

    en_range = np.arange(16, 21+1, 1, dtype = int)
    en_len = len(en_range)

    num_runs = 200

    for r in tqdm(range(en_len)):
        for s in tqdm(range(num_runs)):
            statements = get_dag_statement(station, en_range[r], s)
            with open(dag_file_name, 'a') as f:
                f.write(statements)
            del statements

    print('Done!')

if __name__ == "__main__":

    main()

 
