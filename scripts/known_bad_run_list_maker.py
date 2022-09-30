import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

def get_bad_list(bad_path):
    bad_run_arr = []
    with open(bad_path, 'r') as f:
        for lines in f:
            run_num = int(lines)
            bad_run_arr.append(run_num)
    bad_run_arr = np.asarray(bad_run_arr, dtype = int)
    return bad_run_arr

k_path = f'../data/known_runs/'
if not os.path.exists(k_path):
    os.makedirs(k_path)
k_name = f'{k_path}known_runs_A{Station}.txt'

knwon_issue = known_issue_loader(Station)

bad_surface_run = knwon_issue.get_bad_surface_run()
bad_run = knwon_issue.get_bad_run()
knwon_bad_run = np.append(bad_surface_run, bad_run)
L0_to_L1_Processing = knwon_issue.get_L0_to_L1_Processing_run()
knwon_bad_run = np.append(knwon_bad_run, L0_to_L1_Processing)
ARARunLogDataBase = knwon_issue.get_ARARunLogDataBase()
knwon_bad_run = np.append(knwon_bad_run, ARARunLogDataBase)
software_dominant_run = knwon_issue.get_software_dominant_run()
knwon_bad_run = np.append(knwon_bad_run, software_dominant_run)
ob_bad_run = knwon_issue.get_obviously_bad_run()
knwon_bad_run = np.append(knwon_bad_run, ob_bad_run)

q_path = f'../data/qual_runs/qual_run_A{Station}.txt'
q_bad_run_arr = []
with open(q_path, 'r') as f:
    for lines in f:
        run_num = int(lines.split()[0])
        q_bad_run_arr.append(run_num)
q_bad_run_arr = np.asarray(q_bad_run_arr, dtype = int)
knwon_bad_run = np.append(knwon_bad_run, q_bad_run_arr)

r_path = f'../data/rayl_runs/rayl_run_A{Station}.txt'
r_bad_run_arr = []
with open(r_path, 'r') as f:
    for lines in f:
        run_num = int(lines)
        r_bad_run_arr.append(run_num)
r_bad_run_arr = np.asarray(r_bad_run_arr, dtype = int)
knwon_bad_run = np.append(knwon_bad_run, r_bad_run_arr)

knwon_bad_run = np.unique(knwon_bad_run).astype(int)
print(knwon_bad_run)
print(len(knwon_bad_run))
print(knwon_bad_run.dtype)

np.savetxt(k_name, knwon_bad_run)
print(k_name)
print('done!')
