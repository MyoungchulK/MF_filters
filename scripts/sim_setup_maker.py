import os, sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import click
import h5py
import csv
from scipy.interpolate import interp1d

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

def get_dag_statement(st, run, flavors = None):

    statements = ""
    if flavors is not None:
        flavors_int = int(flavors)
        statements += f'JOB job_ARA_F{flavors_int}_S{st}_R{run} ARA_job.sub \n'
        statements += f'VARS job_ARA_F{flavors_int}_S{st}_R{run} fla="{flavors_int}" st="{st}" run="{run}"\n\n'
    else:
        statements += f'JOB job_ARA_S{st}_R{run} ARA_job.sub \n'
        statements += f'VARS job_ARA_S{st}_R{run} st="{st}" run="{run}"\n\n'

    return statements

@click.command()
@click.option('-k', '--key', type = str)
@click.option('-s', '--station', type = int)
@click.option('-b', '--blind_dat', default = False, type = bool)
def main(key, station, blind_dat):
    
    st_old = 'DETECTOR_STATION='
    config_old = 'DETECTOR_STATION_LIVETIME_CONFIG='
    run_old = 'DETECTOR_RUN='
    trig_old = 'TRIG_WINDOW='
    wf_len_old = 'WAVEFORM_LENGTH='
    thres_old = 'POWERTHRESHOLD='
    ele_old = 'CUSTOM_ELECTRONICS='
    if key == 'signal':
        flavor_old = 'SELECT_FLAVOR='

    blind = ''
    if blind_dat:
        blind = '_full'
    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/rayl{blind}/' # rayl path
    d_list, d_run_tot, d_run_range = file_sorter(d_path+'*h5')
    e_path = f'../sim/sim_setup_example/{key}_rayl.txt' # setup ex path
    r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/sim_{key}_setup{blind}/' # output path
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    if key == 'noise':
        t_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/sim_table{blind}/' # table output path
        if not os.path.exists(t_path):
            os.makedirs(t_path)
    dag_path = f'/home/mkim/analysis/MF_filters/scripts/batch_run/wipac_sim_{key}/' # dag path
    if not os.path.exists(dag_path):
        os.makedirs(dag_path)
    dag_file_name = f'{dag_path}A{station}.dag'
    statements = ""
    with open(dag_file_name, 'w') as f:
        f.write(statements)
    del d_run_range

    if key == 'noise':
        dt = 0.5
        freq_mhz = np.fft.rfftfreq(2280, 0.5) * 1e3
        fft_len = len(freq_mhz)
        pahse = np.loadtxt(f'../data/sc_info/SC_Phase_from_sim.txt')
        p_f = interp1d(pahse[:,0], pahse[:, 1:], axis = 0, fill_value = 'extrapolate')
        pahse_int = p_f(freq_mhz)
        h_tot = np.loadtxt(f'../data/sc_info/A{station}_Htot.txt')
        h_f = interp1d(h_tot[:,0], h_tot[:, 1:], axis = 0, fill_value = 'extrapolate')
        h_tot_int = h_f(freq_mhz)
        Htot = h_tot_int * np.sqrt(dt * 1e-9)
        
        sc_hf = h5py.File(f'../data/sc_info/A{station}_sc_tot.h5', 'r')
        freq = sc_hf['freq_range'][:]
        sc = sc_hf['sc'][:] 
        print(sc.shape)
        s_f =  interp1d(freq, sc, axis = 0, fill_value = 'extrapolate')
        sc_tot = s_f(freq_mhz)
    
        r_hf = h5py.File(f'../data/sc_info/A{station}_rayl_tot.h5', 'r')
        r_freq = r_hf['freq_range'][:]
        rayls = r_hf['rayl'][:]
        print(rayls.shape)
        r_f =  interp1d(r_freq, rayls, axis = 0, fill_value = 'extrapolate')
        rayl_tot = r_f(freq_mhz)
        del sc_hf, pahse, p_f, h_tot, h_f, h_tot_int, dt, freq, sc, s_f, r_hf, r_freq, rayls, r_f

    for r in tqdm(range(len(d_run_tot))):
      #if d_run_tot[r] == 12001:
        #if station == 3 and d_run_tot[r] == 3429:
        #    continue

        hf = h5py.File(d_list[r], 'r')
        soft_len = hf['soft_len'][:]   
        rf_len = hf['rf_len'][:]
        if soft_len.shape[-1] == 0 or len(rf_len) == 0:
            print(d_list[r], 'empty!!!')
            continue
        soft_rayl = hf['soft_rayl'][:]
        soft_sc = hf['soft_sc'][:]
        if np.any(np.isnan(soft_rayl.flatten())) or np.any(np.isnan(soft_sc.flatten())):
            print(d_list[r], 'nan!!!')
            continue
        del soft_sc, soft_len

        if key == 'noise':
            p1 = np.nansum(soft_rayl, axis = 0) / 1e3 * np.sqrt(1e-9)
        del soft_rayl
        unix_time = hf['unix_time'][0]
        year = int(datetime.utcfromtimestamp(unix_time).strftime('%Y%m%d%H%M%S')[:4])
        wf_len = np.nanmedian(rf_len.flatten()).astype(int)
        del hf, unix_time, rf_len
 
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
            del thres_val
        ele_new = f'{ele_old}3'
        del wf_len, year, trig_val

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
                for f in range(3):    
                    fla_idx_old = int(f)
                    fla_idx = int(fla_idx_old + 1)
                    flavor_old_temp = f'{flavor_old}{fla_idx_old}'
                    flavor_new = f'{flavor_old}{fla_idx}'
                    context = context.replace(flavor_old_temp, flavor_new)
                    
                    n_path = f'{r_path}{key}_F{fla_idx}_A{station}_R{d_run_tot[r]}.txt'
                    with open(n_path, "w") as f:
                        f.write(context)   

                    statements = get_dag_statement(station, d_run_tot[r], flavors = fla_idx)
                    with open(dag_file_name, 'a') as f:
                        f.write(statements)
                    del statements

        if key == 'noise': 
            n_path = f'{r_path}{key}_A{station}_R{d_run_tot[r]}.txt'
            with open(n_path, "w") as f:
                f.write(context)

            statements = get_dag_statement(station, d_run_tot[r])
            with open(dag_file_name, 'a') as f:
                f.write(statements)
            del statements

            # csv
            csv_name = f'{t_path}rayl{blind}_A{station}_R{d_run_tot[r]}.csv'
            with open(csv_name, 'w', newline='') as rayl_file:
                writer = csv.writer(rayl_file)
                writer.writerow(["freqs[MHz]","channel","p1[V/(sqrt(Hz))]", "chi2[TBA]"])
                for freq in range(fft_len):
                    for ant in range(16):
                        if station == 3 and ant % 4 == 0 and config == 7:
                            p1_r = rayl_tot[freq, ant, 5]
                        elif station == 3 and ant % 4 == 3 and (config == 3 or config == 4 or config == 5):
                            p1_r = rayl_tot[freq, ant, 1]
                        else:
                            p1_r = p1[freq, ant]

                        writer.writerow([freq_mhz[freq], ant, p1_r, 0])

            # sc
            Hmeas = p1 * np.sqrt(2) * np.sqrt(2)
            soft_sc = Hmeas / Htot
            sc_table = np.full((fft_len, 2 * 16 + 1), np.nan, dtype = float)
            sc_table[:,0] = freq_mhz
            for ant in range(16):

                if station == 3 and ant % 4 == 0 and config == 7:
                    sc_gain = sc_tot[:, ant, 5]
                elif station == 3 and ant % 4 == 3 and (config == 3 or config == 4 or config == 5):
                    sc_gain = sc_tot[:, ant, 1]
                else:
                    sc_gain = soft_sc[:, ant]

                # BV-BH-TV-TH
                if ant == 0: # d1tv
                    new_ant = 14 # s3 tv
                if ant == 4: # d1bv
                    new_ant = 12 # s3 bv
                if ant == 8: # d1th
                    new_ant = 15 # s3 th
                if ant == 12: # d1bh
                   new_ant = 13 # s3 bh
                if ant == 1: # d2tv
                    new_ant = 2 # s0 tv
                if ant == 5: # d2bv
                    new_ant = 0 # s0 bv
                if ant == 9: # d2th
                    new_ant = 3 # s0 th
                if ant == 13: # d2bh
                    new_ant = 1 # s0 bh
                if ant == 2: # d3tv
                    new_ant = 6 # s1 tv
                if ant == 6: # d3bv
                    new_ant = 4 # s1 bv
                if ant == 10: # d3th
                    new_ant = 7 # s1 th
                if ant == 14: # d3bh
                    new_ant = 5 # s1 bh
                if ant == 3: # d4tv
                    new_ant = 10 # s2 tv
                if ant == 7: # d4bv
                    new_ant = 8 # s2 bv
                if ant == 11: # d4th
                    new_ant = 11 # s2 th
                if ant == 15: # d4bh
                    new_ant = 9 # s2 bh

                sc_table[:, 2 * new_ant + 1] = sc_gain
                sc_table[:, 2 * new_ant + 2] = pahse_int[:, new_ant]
            sc_name = f'{t_path}sc{blind}_A{station}_R{d_run_tot[r]}.txt'
            np.savetxt(sc_name, sc_table)
            del Hmeas, soft_sc, sc_name
        del config, st_new, config_new, run_new, trig_new, wf_len_new, thres_new, ele_new

    print('Done!')

if __name__ == "__main__":

    main()

 
