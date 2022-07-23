import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call
from glob import glob

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter

def cobalt_run_loader(Station = None, Key = None, Act_Evt = None, analyze_blind_dat = False):

    print('cobalt run starts!')
    print('event range:', Act_Evt)

    b_type = ''
    if analyze_blind_dat:
        b_type = '_full'

    # sort
    #d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}{b_type}/*'
    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw/*'
    print(d_path)
    lists, d_run_tot, d_run_range = file_sorter(d_path)
    print(d_run_tot)   
 

    #print(lists[652])
    #print(lists[667])

    count = 0
    for w in tqdm(d_run_tot):

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
 
