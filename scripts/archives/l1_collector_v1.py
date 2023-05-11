import numpy as np
import os, sys
from tqdm import tqdm
import h5py
from glob import glob
import re

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

def get_context(p, mask_key, end_key):
    maks_key_len = len(mask_key)
    val_arr = np.full((16), np.nan, dtype = float)

    with open(p,'r') as p_file:
        p_read = p_file.read()

        key_idxs = np.asarray([i.start() for i in re.finditer(mask_key, p_read)]).astype(int)
        key_idxs_mask = np.asarray([i.start() for i in re.finditer('//'+mask_key, p_read)]).astype(int) + 2
        key_idxs_bool = np.in1d(key_idxs, key_idxs_mask, invert = True)
        key_idx = key_idxs[key_idxs_bool]
        if len(key_idx) == 1:
            key_idx_f = key_idx[0] + maks_key_len
            end_key_idx = p_read.find(end_key, key_idx_f)
            val_arr[:] = np.asarray(p_read[key_idx_f:end_key_idx].split(",")).astype(int)

    return val_arr

def l1_loader(Station = None):

    print('Collecting l1 starts!')

    mask_key = 'scalerGoalValues#I16='
    end_key = ';'

    arr_len = 20000
    run_arr = np.full((arr_len), np.nan, dtype = float)
    run_type = np.copy(run_arr)
    bad_type = np.copy(run_arr)
    l1_goal = np.full((arr_len, 16), np.nan, dtype = float)
    
    knwon_issue = known_issue_loader(Station)
    bad_runs = knwon_issue.get_knwon_bad_run()
    bad_runs = np.sort(np.unique(bad_runs))
    if Station == 3:
        bad_runs = bad_runs[:-1]
    bad_type[bad_runs] = 1 
    del knwon_issue, bad_runs

    print('pedestal!')
    batch_info = batch_info_loader(Station)
    yrs_arr = batch_info.years
    for yrs in tqdm(yrs_arr):
        if int(yrs) == 2013:
            continue
        ped_path = f'/data/exp/ARA/{int(yrs)}/calibration/pedestals/ARA0{Station}/'    
        ped_config_path = glob(f'{ped_path}configFile*')
        for p in ped_config_path:
            run_num = int(re.sub("\D", "", p[-10:-4]))
            run_arr[run_num] = run_num
            run_type[run_num] = 0    

            l1_goal_num  = get_context(p, mask_key, end_key)
            if np.all(np.isnan(l1_goal_num)):
                pass
            else:
                l1_goal[run_num] = l1_goal_num
            del run_num, l1_goal_num
        del ped_path, ped_config_path
    del yrs_arr

    print('unblined event!')
    run_num, evt_path = batch_info.get_dat_list(analyze_blind_dat = False)[:2]
    run_num = run_num.astype(int)
    run_arr[run_num] = run_num
    run_type[run_num] = 1

    for run in tqdm(range(len(run_num))):
        slash_idx = evt_path[run].rfind('/')
        run_path = evt_path[run][:slash_idx] + '/'
        run_config_path = glob(f'{run_path}configFile*')
        if len(run_config_path) != 1:
            print(os.listdir(run_path))
            continue

        l1_goal_num  = get_context(run_config_path[0], mask_key, end_key)
        if np.all(np.isnan(l1_goal_num)):
            pass
        else:
            l1_goal[run_num[run]] = l1_goal_num
        del slash_idx, run_path, run_config_path, l1_goal_num 
    del run_num, evt_path

    print('blinded event!')
    run_num, evt_path = batch_info.get_dat_list(analyze_blind_dat = True)[:2]
    run_num = run_num.astype(int)
    run_arr[run_num] = run_num
    run_type[run_num] = 1

    for run in tqdm(range(len(run_num))):
        slash_idx = evt_path[run].rfind('/')
        run_path = evt_path[run][:slash_idx] + '/'
        run_config_path = glob(f'{run_path}configFile*')
        if len(run_config_path) != 1:
            print(os.listdir(run_path))
            continue
    
        l1_goal_num  = get_context(run_config_path[0], mask_key, end_key)
        if np.all(np.isnan(l1_goal_num)):
            pass
        else:
            l1_goal[run_num[run]] = l1_goal_num       
        del slash_idx, run_path, run_config_path, l1_goal_num
    del run_num, evt_path, batch_info

    print('L1 collecting is done!')

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}L1_Goal_A{Station}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('run_type', data=run_type, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_type', data=bad_type, compression="gzip", compression_opts=9)
    hf.create_dataset('l1_goal', data=l1_goal, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)

if __name__ == "__main__":

    if len (sys.argv) != 2:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    
    l1_loader(Station = station)
 
