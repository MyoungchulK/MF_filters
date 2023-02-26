import os, sys
import numpy as np
from tqdm import tqdm
import click

def get_trig_win_len(sation, config):
    config_idx = int(config - 1)
    if sation == 2:
        trig_win = np.array([1.1, 1.1, 1.1, 1.7, 1.7, 1.7, 1.7], dtype = float)
    else:
        trig_win = np.array([1.1, 1.1, 1.7, 1.7, 1.1, 1.7, 1.7, 1.7, 1.7], dtype = float)
    trig_sel = trig_win[config_idx]
    return trig_sel

def get_wf_len(station, config):
    config_idx = int(config - 1)
    if station == 2:
        wf_len = np.array([800, 800, 800, 1040, 1040, 1120, 1120], dtype = int)
    else:
        wf_len = np.array([800, 800, 800, 1040, 800, 1120, 1120, 1120, 1120], dtype = int)
    wf_sel = int(wf_len[config_idx] + 40)
    return wf_sel

def get_year(station, config):
    config_idx = int(config - 1)
    if station == 2:
        yr = np.array([2014, 2013, 2014, 2015, 2016, 2017, 2020], dtype = int)
    else:
        yr = np.array([2013, 2013, 2015, 2016, 2014, 2018, 2019, 2019, 2020], dtype = int)
    yr_sel = yr[config_idx]
    return yr_sel

def get_thres(station, year):
    year_idx = int(year - 2013)
    if station == 2:
        thres_arr = np.array([-6.428, -6.428, -6.6, -6.6, -6.603, -6.6, -6.6, -6.6], dtype = float)
    else:
        thres_arr = np.array([-6.43, -6.43, -6.6, -6.6, -6.608, -6.608, -6.608, -6.608], dtype = float)
    thres_sel = thres_arr[year_idx]
    return thres_sel

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
    
    e_path = f'../sim/sim_setup_example/{key}_rayl.txt' # setup ex path
    blind = ''
    if blind_dat:
        blind = '_full'
    r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/sim_{key}_setup{blind}/' # text output path
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{key}_config/' # dag path
    if not os.path.exists(dag_path):
        os.makedirs(dag_path)
    dag_file_name = f'{dag_path}A{station}.dag'
    statements = ""
    with open(dag_file_name, 'w') as f:
        f.write(statements)

    st_old = 'DETECTOR_STATION='
    config_old = 'DETECTOR_STATION_LIVETIME_CONFIG='
    run_old = 'DETECTOR_RUN='
    trig_old = 'TRIG_WINDOW='
    wf_len_old = 'WAVEFORM_LENGTH='
    thres_old = 'POWERTHRESHOLD='
    ele_old = 'CUSTOM_ELECTRONICS='
    if key == 'signal':
        flavor_old = 'SELECT_FLAVOR='

    if station == 2:
        num_configs = 7
    if station == 3:
        num_configs = 9
    if key == 'signal':
        num_evts = 10
        num_runs = 3000
    else:
        num_evts = 1000
        num_runs = 1000

    for r in tqdm(range(num_configs)):

        st_new = f'{st_old}{station}'
        config_new = f'{config_old}{int(r+1)}'
        run_new = f'{run_old}{int(r+1)}'
        ele_new = f'{ele_old}3'
        trig_val = get_trig_win_len(station, int(r+1))
        trig_new = f'{trig_old}{trig_val}E-7'
        wf_len = get_wf_len(station, int(r+1))
        wf_len_new = f'{wf_len_old}{wf_len}'

        if key == 'noise':
            thres_new = f'{thres_old}-3'
        else:
            year = get_year(station, int(r+1))
            thres_val = get_thres(station, year)
            thres_new = f'{thres_old}{thres_val}'

        with open(e_path, "r") as f:
            context = f.read()
            context = context.replace(st_old, st_new)
            context = context.replace(config_old, config_new)
            context = context.replace(run_old, run_new)
            context = context.replace(trig_old, trig_new)
            context = context.replace(wf_len_old, wf_len_new)
            context = context.replace(thres_old, thres_new)
            context = context.replace(ele_old, ele_new)
            if key == 'signal':
                nnu_pass = 'NNU_PASSED=5'
            else:
                nnu_pass = 'NNU_PASSED=500'
            nnu_pass_new = f'NNU_PASSED={num_evts}'
            context = context.replace(nnu_pass, nnu_pass_new)
            if key == 'signal':
                for f in range(3):
                    fla_idx_old = int(f)
                    fla_idx = int(fla_idx_old + 1)
                    flavor_old_temp = f'{flavor_old}{fla_idx_old}'
                    flavor_new = f'{flavor_old}{fla_idx}'
                    context = context.replace(flavor_old_temp, flavor_new)

                    n_path = f'{r_path}{key}_F{fla_idx}_A{station}_R{int(r+1)}.txt'
                    with open(n_path, "w") as f:
                        f.write(context)

                    for s in tqdm(range(num_runs)):
                        statements = get_dag_statement(station, int(r+1), s, flavors = fla_idx)
                        with open(dag_file_name, 'a') as f:
                            f.write(statements)
                        del statements

            if key == 'noise':
                n_path = f'{r_path}{key}_A{station}_R{int(r+1)}.txt'
                with open(n_path, "w") as f:
                    f.write(context)

                for s in tqdm(range(num_runs)):
                    statements = get_dag_statement(station, int(r+1), s)
                    with open(dag_file_name, 'a') as f:
                        f.write(statements)


    print('Done!')

if __name__ == "__main__":

    main()

 
