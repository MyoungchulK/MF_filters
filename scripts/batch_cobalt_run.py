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

    count = 0
    for w in tqdm(lists):

        if count >= Act_Evt[0] and count < Act_Evt[1]:

            CMD_line = f'python3 -W ignore script_executor.py -k {Key} -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1 -l 1'
            print(CMD_line)
            call(CMD_line.split(' '))

            #run_info = run_info_loader(Station, int(w), analyze_blind_dat = analyze_blind_dat)
            #daq_dat = run_info.get_result_path(file_type = Key, verbose = True)
            #if os.path.exists(daq_dat):
            #    print(f'{daq_dat} is already there!!')
            #else:
            #    print(f'{count} THERE IS NO {daq_dat}!!')
            #    break
            #    return

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
 
