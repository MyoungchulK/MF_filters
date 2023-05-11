import numpy as np
import os, sys
from tqdm import tqdm
from subprocess import call

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

def cobalt_run_loader(Station = None, Act_Evt = None):

    print('cobalt run starts!')
    print('event range:', Act_Evt)

    list_path = '../data/run_list/'
    list_name = f'{list_path}A{Station}_run_list_full.txt'
    list_file =  open(list_name, "r")
    lists = []
    path_lists = []
    for lines in list_file:
        line = lines.split()
        run_num = int(line[0])
        lists.append(run_num)
        path_lists.append(line[1])
        del line
    list_file.close()
    del list_path, list_name, list_file
    lists = np.asarray(lists, dtype = int)

    ped_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/ped_full/'
    new_ped_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/ped_full_new/'

    for count in tqdm(range(len(lists))):

        if count >= Act_Evt[0] and count < Act_Evt[1]:

            qual = f'{ped_path}ped_full_qualities_A{Station}_R{lists[count]}.dat'
            out = f'{new_ped_path}ped_full_values_A{Station}_R{lists[count]}.dat'

            CMD_line = f'/home/mkim/analysis/AraSoft/AraUtil/bin/repeder -d -m 0 -M 4096 -q {qual} {path_lists[count]} {out}'
            print(CMD_line)
            call(CMD_line.split(' '))


    print('cobalt run is done!')

if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Event # ex)-1,1000000>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    act_evt = np.array([-1,1000000], dtype = int)
    if len(sys.argv) > 2:
        act_evt = np.asarray(sys.argv[2].split(','), dtype = int)

    cobalt_run_loader(Station = station, Act_Evt = act_evt)
 
