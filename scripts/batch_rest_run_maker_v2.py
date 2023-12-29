import numpy as np
import os, sys
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader
from tools.ara_run_manager import file_sorter

def batch_run_loader(Station = None, Input = None, Output = None, Analyze_Blind = False):

    batch_info = batch_info_loader(Station)

    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    blind_type = ''
    if Analyze_Blind:
        blind_type = '_full'
    list_path = '../data/run_list/'
    list_name = f'{list_path}A{Station}_run_list{blind_type}.txt'
    list_file =  open(list_name, "r")
    lists = []
    for lines in list_file:
        line = lines.split()
        run_num = int(line[0])
        lists.append(run_num)
        del line
    list_file.close()
    del list_path, list_name, list_file
    lists = np.asarray(lists, dtype = int)

    new_list = []

    for w in tqdm(lists):

            if Station == 3 and int(w) == 482: continue

            log_name = f'{Input}A{Station}.R{int(w)}.log'
            err_name = f'{Input}A{Station}.R{int(w)}.err'
            log_flag = False
            err_flag = False
            if not os.path.exists(log_name) and not os.path.exists(err_name): pass
            else:
                if os.path.exists(log_name):
                    with open(log_name,'r') as f:
                        f_read = f.read()
                        key_idx = f_read.find('Error')
                        if key_idx != -1:
                            log_flag = True
                if os.path.exists(err_name):
                    with open(err_name,'r') as f:
                        f_read = f.read()
                        key_idx = f_read.find('Error')
                        if key_idx != -1:
                            err_flag = True
                if log_flag == False and err_flag == False: pass
                else:
                    new_list.append(int(w))
            if Station == 3 and os.path.exists(log_name):
                log_flag = False
                with open(log_name,'r') as f:
                    f_read = f.read()
                    key_idx = f_read[-300:].find('aborted')
                    key_idx1 = f_read[-300:].find('2023-12-28')
                    if key_idx != -1 and key_idx1 != -1:
                        log_flag = True
                if log_flag: 
                    print(int(w))
                    new_list.append(int(w))
            if Station == 2 and os.path.exists(log_name):
                log_flag = False
                with open(log_name,'r') as f:
                    f_read = f.read()
                    key_idx = f_read[-300:].find('aborted')
                    key_idx1 = f_read[-300:].find('2023-12-29')
                    if key_idx != -1 and key_idx1 != -1:
                        log_flag = True
                if log_flag:
                    print(int(w))
                    new_list.append(int(w))

    new_list = np.asarray(new_list, dtype = int)
    new_list = np.unique(new_list).astype(int)
    print('!!!!!!all runs!!!!!!', len(lists))
    print('!!!!!!error runs!!!!!!', len(new_list))

    d_idx = ~np.in1d(lists, new_list)
    d_run_tot = lists[d_idx]
    print('!!!!!!rest runs!!!!!!', len(d_run_tot))

    bad_path = f'/home/mkim/analysis/MF_filters/data/run_list/A{Station}_run_list{blind_type}.txt'
    print(bad_path)
    batch_info.get_rest_dag_file_v2(Output, d_run_tot, bad_path)    

if __name__ == "__main__":

    if len (sys.argv) < 5:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Input ex)/home/mkim/wipac_full/logs/>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>
    <Analyze_Blind ex)0 or 1>
    <Key ex)snr>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    inputs = str(sys.argv[2])
    output=str(sys.argv[3])
    blind=bool(int(sys.argv[4]))

    batch_run_loader(Station = station, Input = inputs, Output = output, Analyze_Blind = blind)
 
