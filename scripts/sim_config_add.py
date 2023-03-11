import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_run_manager import get_path_info_v2

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

for r in tqdm(range(len(d_run_tot))):
    
  #if r == 1:

    data = d_list[r]
    config = int(get_path_info_v2(data, '_R', '.txt'))
    flavor = int(get_path_info_v2(data, 'AraOut.signal_F', '_A'))
    sim_run = int(get_path_info_v2(data, 'txt.run', '.h5'))
    if config < 6:
        year = 2015
    else:
        year = 2018

    cons = np.array([Station, sim_run, config, year, flavor])

    hf = h5py.File(data, 'r+')
    try:
        #tree_r = hf['config'][:]
        #print(tree_r)
        del hf['config']
    except KeyError:
        pass
    hf.create_dataset('config', data=cons, compression="gzip", compression_opts=9)
    hf.close()    

    #hf = h5py.File(data, 'r')
    #tree_r = hf['config'][:]
    #print(tree_r)
    #hf.close()

print('done!')





