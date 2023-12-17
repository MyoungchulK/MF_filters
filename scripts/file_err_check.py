import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

Station = int(sys.argv[1])
Name = str(sys.argv[2])
Tree = str(sys.argv[3])
Blind = int(sys.argv[4])
if Blind: 
    b_name = '_full'
    bb_name = '_Full'
else: 
    b_name = ''
    bb_name = ''

# sort
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/{Name}{b_name}/'

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary{bb_name}_A{Station}.h5'
hf = h5py.File(r_path + file_name, 'r')
runs = hf['runs'][:]
num_runs = len(runs)
del hf, r_path, file_name

batch_info = batch_info_loader(Station)

bad_run = []

for r in tqdm(range(num_runs)):
    
  #if r <10:

    m_name = f'{m_path}{Name}{b_name}_A{Station}_R{runs[r]}.h5'
    
    if not os.path.exists(m_name): continue

    try:
        hf = h5py.File(m_name, 'r')
        lists = list(hf)
        tree = hf[Tree][:]
    except Exception as e:
        print(e, m_name)
        bad_run.append(runs[r])

bad_run = np.asarray(bad_run).astype(int)
d_idx = ~np.in1d(runs, bad_run)
d_run_tot = runs[d_idx]

#bad_path = f'/home/mkim/analysis/MF_filters/data/run_list/A{Station}_run_list{b_name}.txt'
#batch_info.get_rest_dag_file_v2('/home/mkim/', d_run_tot, bad_path)

print(bad_run)
print(len(bad_run)) 
print('done!')





