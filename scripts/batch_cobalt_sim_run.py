import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call
from glob import glob

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def cobalt_run_loader(Key = None, Station = None, Act_Evt = None):

    print('cobalt run starts!')
    print('event range:', Act_Evt)
    Yrs = 2015
  
    #d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sim_signal_full/AraOut*root'
    #d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sim_noise_full/AraOut*root'
    d_path = f'/misc/disk19/users/mkim/OMF_filter/ARA0{Station}/sim_signal_full/AraOut*root'
    #d_path = f'/misc/disk20/users/mkim/OMF_filter/ARA0{Station}/sim_noise_full/AraOut*root'
    lists = glob(d_path)
    print(len(lists))
 
    count = 0
    for w in tqdm(lists):

        if count >= Act_Evt[0] and count < Act_Evt[1]:

            CMD_line = f'python3 -W ignore sim_script_executor.py -k {Key} -s {Station} -y {Yrs} -d {w} -n 1'
            print(CMD_line)
            call(CMD_line.split(' '))
        
        count += 1

    print('cobalt run is done!')

if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <key ex)sensor>
    <Station ex)2>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key = str(sys.argv[1])
    station=int(sys.argv[2])
    act_evt = np.array([-1,1000000], dtype = int)
    if len(sys.argv) > 3:
        act_evt = np.asarray(sys.argv[3].split(','), dtype = int)

    cobalt_run_loader(Key = key, Station = station, Act_Evt = act_evt)
 
