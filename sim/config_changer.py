import numpy as np
import os, sys
from tqdm import tqdm
from glob import glob

#d_path = f'/data/user/mkim/OMF_filter/ARA02/sim_noise_setup/*txt'
d_path = f'/misc/disk19/users/mkim/OMF_filter/ARA03/sim_noise_setup/*txt'
d_list = glob(d_path)

old_line = 'NNU_PASSED=500'
new_line = 'NNU_PASSED=100'

for t in tqdm(range(len(d_list))):

    #if t != 0:
    #    continue
    #print(d_list[t])

    with open(d_list[t], "r") as f:    
        context = f.read()
        context = context.replace(old_line, new_line)

    with open(d_list[t], "w") as f:        
        f.write(context)
 

