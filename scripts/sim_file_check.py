import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
import uproot

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)

for r in tqdm(range(len(d_run_tot))):

  if r > 1140:

    try: 
        file_uproot = uproot.open(d_list[r])   
    except ValueError:
        print(d_list[r])
        continue
    pnu = np.asarray(file_uproot['AraTree2/event/pnu'], dtype = float)
    if len(pnu) != 100:
        print(d_list[r])
    del file_uproot, pnu

 
