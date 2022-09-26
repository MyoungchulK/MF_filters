import os, sys
import numpy as np
from tqdm import tqdm
import click
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter

@click.command()
@click.option('-s', '--station', type = int)
def main(station):
    
    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/rayl_full/' # rayl path
    d_list, d_run_tot, d_run_range = file_sorter(d_path+'*h5')

    bad_path = f'../data/rayl_runs/rayl_run_A{station}.txt'
    bad_run_arr = []
    with open(bad_path, 'r') as f:
        for lines in f:
            run_num = int(lines)
            bad_run_arr.append(run_num)
    bad_run_arr = np.asarray(bad_run_arr, dtype = int)

    for r in tqdm(range(len(d_run_tot))):
        
        hf = h5py.File(d_list[r], 'r')
        soft_len = hf['soft_len'][:]   
        soft_rayl = hf['soft_rayl'][:]
        evt_len = soft_len.shape[-1]
        nan_table = np.any(np.isnan(soft_rayl.flatten()))
        if evt_len == 0 or nan_table:
            if d_run_tot[r] in bad_run_arr:
                pass
            else:
                print(f'WTF A{station} R{d_run_tot[r]} length:{evt_len} nan:{nan_table} !!!!!!!!!!!!!!!!!!!')
        del hf, soft_len, soft_rayl, evt_len, nan_table

    print('Done!')

if __name__ == "__main__":

    main()

 
