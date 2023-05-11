import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

known_issue = known_issue_loader(Station)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
print(bad_runs)
print(f'# of bad runs: {len(bad_runs)}')

runs = np.loadtxt(f'/home/mkim/A{Station}.txt')
runs = runs.astype(int)
print(len(runs))

idxs = ~np.in1d(runs, bad_runs)
print(np.any(idxs))
print(runs[idxs])
print(len(runs[idxs]))
print('done!')

