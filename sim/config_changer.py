import numpy as np
import os, sys
from tqdm import tqdm
from glob import glob

d_path = f'/home/mkim/analysis/MF_filters/sim/sim_signal/*'
d_list = glob(d_path)

old_line = 'POSNU_RADIUS=10000'
new_line = 'POSNU_RADIUS=5000'

for t in tqdm(range(len(d_list))):

    #if t != 0:
    #    continue
    #print(d_list[t])

    with open(d_list[t], "r") as f:    
        context = f.read()
        context = context.replace(old_line, new_line)

    with open(d_list[t], "w") as f:        
        f.write(context)
 

