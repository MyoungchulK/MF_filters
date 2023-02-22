import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call
from glob import glob

def batch_run_loader(Station = None, Output = None, Analyze_Blind = False, Key = None):

    lists = glob(f'/home/mkim/logs/*')

    new_st = []
    new_list = []

    for w in tqdm(lists):

        #print(w)

        log_flag = False
        with open(w,'r') as f:
            f_read = f.read()
            key_idx = f_read.find('Error')
            if key_idx != -1:
                log_flag = True

        if log_flag:
            i_key = 'A'
            i_key_len = len(i_key)
            i_idx = w.find(i_key)
            f_idx = w.find('.', i_idx + i_key_len)
            st = int(w[i_idx + i_key_len:f_idx])
            new_st.append(st)

            i_key = 'R'
            i_key_len = len(i_key)
            i_idx = w.find(i_key)
            f_idx = w.find('.', i_idx + i_key_len)
            run = int(w[i_idx + i_key_len:f_idx])
            new_list.append(run)

            print(st, run)
    new_st = np.asarray(new_st, dtype = int)
    new_list = np.asarray(new_list, dtype = int)
    print('!!!!!!error runs!!!!!!', len(new_list))

    d_path = f'/home/mkim/A{Station}.dag'
    nd_path = f'/home/mkim/A{Station}_new.dag'
    with open(d_path, 'r') as config_file:
        config_read = config_file.read()

        for r in tqdm(range(len(new_list))):
            dag_line = f'JOB job_ARA_S{new_st[r]}_R{new_list[r]} ARA_job.sub'
            new_dag_line = f'JOB job_ARA_S{new_st[r]}_R{new_list[r]} ARA_job_1.sub'

            key_idx = config_read.find(dag_line) # find the index of the key
            if key_idx != -1:
                print(dag_line)
                config_read = config_read.replace(dag_line, new_dag_line)

        with open(nd_path, 'w') as config_file_new:
            config_file_new.write(config_read)


if __name__ == "__main__":

    if len (sys.argv) < 5:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>
    <Analyze_Blind ex)0 or 1>
    <Key ex)snr>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output=str(sys.argv[2])
    blind=bool(int(sys.argv[3]))
    key=str(sys.argv[4])

    batch_run_loader(Station = station, Output = output, Analyze_Blind = blind, Key = key)
 
