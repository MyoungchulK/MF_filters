import numpy as np
import os, sys
from tqdm import tqdm
from glob import glob

#d_path = f'/misc/disk19/users/mkim/OMF_filter/ARA0*/sim_signal_setup_full/*.txt'
d_path = f'/misc/disk19/users/mkim/OMF_filter/ARA0*/sim_noise_setup_full/*.txt'
d_list = glob(d_path)

common = 'ANALYTIC_RAYTRACE_MODE='
old_line = f'{common}1'
new_line = f'{common}0'

old_line = 'RANDOM_MODE=1'
new_line = 'NFOUR=2048\nRANDOM_MODE=1'

for t in tqdm(range(len(d_list))):

    #if t != 0:
    #    continue
    #print(d_list[t])

    with open(d_list[t], "r") as f:    
        context = f.read()
        context = context.replace(old_line, new_line)

    with open(d_list[t], "w") as f:        
        f.write(context)
 

