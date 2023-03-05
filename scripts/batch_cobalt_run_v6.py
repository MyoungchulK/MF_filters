#!/bin/bash

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

    r_key = 'run="'
    r_key_len = len(r_key)
    e_key = '"'

    list_name = f'/home/mkim/A{Station}.dag'
    list_file =  open(list_name, "r")
    lists = []
    with open(list_name, 'r') as f:
        for lines in f:
            key_idx = lines.find(r_key) # find the index of the key
            if key_idx != -1:
                key_idx += r_key_len
                end_key_idx = lines.find(e_key, key_idx)
                run_num = int(lines[key_idx:end_key_idx])
                lists.append(run_num)
    lists = np.asarray(lists, dtype = int)
    lists = lists[::-1]
    print(lists)
    print(len(lists))
    
    count = 0
    for w in tqdm(lists):

        #if int(w) >= Act_Evt[0] and int(w) < Act_Evt[1]:
        if count >= Act_Evt[0] and count < Act_Evt[1]:
            
            CMD_line = f'python3 -W ignore script_executor.py -k {key} -s {Station} -r {int(w)} -b {int(analyze_blind_dat)} -n 1 -q 0'
            print(count)
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
 
