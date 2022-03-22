import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def cobalt_run_loader(Station = None, Key = None, Act_Evt = None, analyze_blind_dat = False):

    print('cobalt run starts!')
    print('event range:', Act_Evt)

    batch_info = batch_info_loader(Station)
    lists = batch_info.get_dat_list(analyze_blind_dat = analyze_blind_dat)[0]

    print(lists[652])
    print(lists[667])

    count = 0
    for w in tqdm(lists):

        if count >= Act_Evt[0] and count < Act_Evt[1]:

            #BASH_line = f'source ../../AraSoft/for_local_araroot.sh'
            #call(BASH_line.split(' '))

            CMD_line = f'python3 -W ignore script_executor.py {Key} {Station} {int(w)} {int(analyze_blind_dat)}'
            print(CMD_line)
            call(CMD_line.split(' '))

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
 
