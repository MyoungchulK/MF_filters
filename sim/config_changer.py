import numpy as np
import os, sys
from tqdm import tqdm
from glob import glob

d_path = f'/home/mkim/analysis/MF_filters/sim/ARA0*/sim_temp_setup_full/*.txt'
#d_path = f'/home/mkim/analysis/MF_filters/sim/ARA0*/sim_signal_setup_full/*.txt'
#d_path = f'/home/mkim/analysis/MF_filters/sim/ARA0*/sim_noise_setup_full/*.txt'
d_list = glob(d_path)

common = 'NNU_PASSED='
old_line = f'{common}15'
new_line = f'{common}12'

old_line = 'DETECTOR_CH_MASK=1'
new_line = 'DETECTOR_CH_MASK=0'
#old_line = 'DETECTOR_TRIG_DELAY=1'
#new_line = 'DETECTOR_TRIG_DELAY=0'
#old_line = 'V_SATURATION=5000'
#new_line = 'V_SATURATION=50000'
#new_line = 'RANDOM_MODE=1\nDETECTOR_CH_MASK=0\nDETECTOR_TRIG_DELAY=0'

for t in tqdm(range(len(d_list))):

    #if t != 0:
    #    continue
    #print(d_list[t])

    with open(d_list[t], "r") as f:    
        context = f.read()
        context = context.replace(old_line, new_line)

    with open(d_list[t], "w") as f:        
        f.write(context)
 

