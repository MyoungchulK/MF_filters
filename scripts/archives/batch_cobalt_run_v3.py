import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader

def cobalt_run_loader(Station = None, Key = None, Act_Evt = None, analyze_blind_dat = False):

    print('cobalt run starts!')
    print('event range:', Act_Evt)

    blind_type = ''
    if analyze_blind_dat:
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
    del list_path, list_name, list_file, blind_type
    lists = np.asarray(lists, dtype = int)

    new_list = []

    for w in tqdm(lists):

            log_name = f'/home/mkim/log_2/A{Station}.R{int(w)}.log'
            err_name = f'/home/mkim/log_2/A{Station}.R{int(w)}.err'
            log_flag = False
            err_flag = False
            if not os.path.exists(log_name) and not os.path.exists(err_name): continue
            
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

            if log_flag == False and err_flag == False: continue

            new_list.append(int(w))
    new_list = np.asarray(new_list, dtype = int)
    print('!!!!!!new runs!!!!!!', len(new_list))

    count = 0
    for w in tqdm(new_list):

        if count >= Act_Evt[0] and count < Act_Evt[1]:
            #if Station == 2 and int(w) == 1451: continue
            if Station == 3 and int(w) == 499: continue
            #if Station == 3 and int(w) == 1966: continue

            CMD_line = f'python3 -W ignore script_executor.py -k qual_cut_1st -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1'
            print(CMD_line)
            call(CMD_line.split(' '))

            CMD_line = f'python3 -W ignore script_executor.py -k baseline -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1'
            print(CMD_line)
            call(CMD_line.split(' '))
            """
            CMD_line = f'python3 -W ignore script_executor.py -k {Key} -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1'
            print(CMD_line)
            call(CMD_line.split(' '))

            CMD_line = f'python3 -W ignore script_executor.py -k cw_ratio -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1'
            print(CMD_line)
            call(CMD_line.split(' '))

            CMD_line = f'python3 -W ignore script_executor.py -k qual_cut -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -q 1'
            print(CMD_line)
            call(CMD_line.split(' '))
            """
        count += 1

    print('cobalt run is done!')

if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <key ex)sensor>
    <blind_type ex)1>
    <Event # ex)-1,1000000>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    key = str(sys.argv[2])
    blind_type = bool(int(sys.argv[3]))
    act_evt = np.array([-1,1000000], dtype = int)
    if len(sys.argv) > 4:
        act_evt = np.asarray(sys.argv[4].split(','), dtype = int)

    cobalt_run_loader(Station = station, Key = key, Act_Evt = act_evt, analyze_blind_dat = blind_type)
 
