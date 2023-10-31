import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
#from subprocess import call
import subprocess

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

Station = int(sys.argv[1])
D_PATH = str(sys.argv[2])

# sort
print('Tar path:', D_PATH)

d_list_chaos = glob(f'{D_PATH}/*')
d_len = len(d_list_chaos)
print('Total Runs:',d_len)

for d in d_list_chaos:

    subprocess.run(f'cd {d}; pwd', shell = True) # file path
    subprocess.run(f'cd {d}; ls -1 | wc -l', shell = True) # file counts
    subprocess.run(f'cd {d}; du -sh', shell = True) # total sizes
