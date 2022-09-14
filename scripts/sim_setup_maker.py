import os, sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import click
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_run_manager import run_info_loader

def get_trig_win_len(sation, config):
    config_idx = int(config - 1)
    if sation == 2:
        trig_win = np.array([1.1, 1.1, 1.1, 1.7, 1.7, 1.7], dtype = float)
    else:
        trig_win = np.array([1.1, 1.1, 1.7, 1.7, 1.1, 1.7, 1.7], dtype = float)
    trig_sel = trig_win[config_idx]
    return trig_sel

def get_thres(station, year):
    year_idx = int(year - 2013)
    if station == 2:
        thres_arr = np.array([-6.428, -6.428, -6.6, -6.6, -6.603, -6.6, -6.6], dtype = float)
    else:
        thres_arr = np.array([-6.43, -6.43, -6.6, -6.6, -1, -6.608, -6.608], dtype = float)
    thres_sel = thres_arr[year_idx]
    return thres_sel

def get_dag_statement(st, run):

    statements = ""
    statements += f'JOB job_ARA_S{st}_R{run} ARA_job.sub \n'
    statements += f'VARS job_ARA_S{st}_R{run} st="{st}" run="{run}"\n\n'

    return statements

@click.command()
@click.option('-k', '--key', type = str)
@click.option('-s', '--station', type = int)
@click.option('-b', '--blind_dat', default = False, type = bool)
def main(key, station, blind_dat):
    
    blind = ''
    if blind_dat:
        blind = '_full'

    # sort
    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/rayl{blind}/*h5'
    d_list, d_run_tot, d_run_range = file_sorter(d_path)
    del d_path, d_run_range
    l_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/rf_len{blind}/'

    e_path = f'../sim/sim_setup_example/{key}_rayl.txt'

    r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/sim_{key}_setup{blind}/'
    if not os.path.exists(r_path):
        os.makedirs(r_path)

    st_old = 'DETECTOR_STATION='
    config_old = 'DETECTOR_STATION_LIVETIME_CONFIG='
    run_old = 'DETECTOR_RUN='
    trig_old = 'TRIG_WINDOW='
    wf_len_old = 'WAVEFORM_LENGTH='
    thres_old = 'POWERTHRESHOLD='
    ele_old = 'CUSTOM_ELECTRONICS='

    dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{key}/'
    if not os.path.exists(dag_path):
        os.makedirs(dag_path)

    dag_file_name = f'{dag_path}A{station}.dag'
    statements = ""
    with open(dag_file_name, 'w') as f:
        f.write(statements)

    for r in tqdm(range(len(d_run_tot))):
       
        hf = h5py.File(d_list[r], 'r')
        soft_len = hf['soft_len'][:]   
       
        try: 
            rf_len = hf['rf_len'][:]
        except KeyError:
            hf_r = h5py.File(f'{l_path}rf_len_A{station}_R{d_run_tot[r]}.h5', 'r')
            rf_len = hf_r['rf_len'][:]
            del hf_r

        unix_time = hf['unix_time'][0]
        year = int(datetime.utcfromtimestamp(unix_time).strftime('%Y%m%d%H%M%S')[:4])
        if soft_len.shape[-1] == 0 or len(rf_len) == 0:
            print(d_list[r], 'empty!!!')
            continue
        wf_len = np.nanmedian(rf_len.flatten()).astype(int)
        del hf, soft_len, unix_time, rf_len
 
        # config 
        ara_run = run_info_loader(station, d_run_tot[r])
        config = ara_run.get_config_number()
        del ara_run

        st_new = f'{st_old}{station}'
        config_new = f'{config_old}{config}'
        run_new = f'{run_old}{d_run_tot[r]}'
        trig_val = get_trig_win_len(station, config)
        trig_new = f'{trig_old}{trig_val}E-7'
        wf_len_new = f'{wf_len_old}{wf_len}'
        if key == 'noise':
            thres_new = f'{thres_old}-3'
        else:
            thres_val = get_thres(station, year)
            thres_new = f'{thres_old}{thres_val}'
        ele_new = f'{ele_old}3'
        del wf_len, config, year

        with open(e_path, "r") as f:
            context = f.read()
            context = context.replace(st_old, st_new)
            context = context.replace(config_old, config_new)
            context = context.replace(run_old, run_new)
            context = context.replace(trig_old, trig_new)
            context = context.replace(wf_len_old, wf_len_new)
            context = context.replace(thres_old, thres_new)
            context = context.replace(ele_old, ele_new)
        
        n_path = f'{r_path}{key}_A{station}_R{d_run_tot[r]}.txt'
        with open(n_path, "w") as f:
            f.write(context)
        del n_path, st_new, config_new, run_new, trig_new, wf_len_new, thres_new

        statements = get_dag_statement(station, d_run_tot[r])
        with open(dag_file_name, 'a') as f:
            f.write(statements)
        del statements

    print('Done!')

if __name__ == "__main__":

    main()

 
