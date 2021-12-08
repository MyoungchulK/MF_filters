import numpy as np
import os, sys
import h5py
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

def sample_idx_table_maker(CPath = curr_path, Station = None, Output = None):

    debug = True

    dpath = '/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraRoot/AraEvent/calib/ATRI/'
    dname = f'araAtriStation{Station}SampleTimingNew.txt'
   
    dda_num = 4
    rf_per_dda = 8
    block_num = 512
    samp_num = 64
    ele_num = dda_num * rf_per_dda
    cap_type = 2

    idx_arr = np.full((samp_num, cap_type, ele_num), np.nan)
    time_arr = np.full((samp_num, cap_type, ele_num), np.nan)
    cap_arr = np.full((cap_type, ele_num), 0, dtype = int)

    with open(dpath+dname,'r') as dtxt:
        line = dtxt.readlines()

        for l in tqdm(range(len(line))):
            each_line = np.asarray(line[l].split(), dtype = float)
            dda = int(each_line[0])
            chan = int(each_line[1])
            cap = int(each_line[2])
            cap_num = int(each_line[3])
            val = each_line[4:]          
 
            ele_ch = dda * rf_per_dda + chan
      
            if l%2 == 0: 
                cap_arr[cap, ele_ch] = cap_num
                idx_arr[val.astype(int), cap, ele_ch] = val
            else:
                t_idx = idx_arr[:, cap, ele_ch]
                t_idx = t_idx[~np.isnan(t_idx)].astype(int)
                time_arr[t_idx, cap, ele_ch] = val

    if debug:
        time_arr_txt = np.copy(time_arr)
        idx_arr_txt = np.copy(idx_arr)
        cap_arr_txt = np.copy(cap_arr)

    if Station == 2:        
        VadjRef = np.array([17562, 17765, 17834, 17526])
        currentVadj = np.array([17795, 17725, 17602, 17676])
        Vadj_corr = 1./(0.9978 - 0.0002*(VadjRef - currentVadj))
        Vadj_corr_32 = np.repeat(Vadj_corr[:,np.newaxis], 8, axis=1)
        Vadj_corr_32 = Vadj_corr_32.flatten()
        time_arr *= Vadj_corr_32[np.newaxis, np.newaxis, :]
    if Station == 3:
        Vadj_corr_32 = np.full((ele_num),1,dtype=int)

    if debug:
        time_arr_Vadj = np.copy(time_arr)

        time_arr_rm_overlap = np.full((samp_num, cap_type, ele_num), np.nan)
        idx_arr_rm_overlap = np.full((samp_num, cap_type, ele_num), np.nan)

    print(cap_arr)
    cap_arr_old = np.copy(cap_arr)

    for b in range(ele_num): 

        even_time = time_arr[:,0,b]
        odd_time = time_arr[:,1,b]

        even_isnan = np.isnan(even_time)
        odd_isnan = np.isnan(odd_time)

        even_time = even_time[~even_isnan]
        odd_time = odd_time[~odd_isnan]

        if debug:
            even_time_trim = np.copy(even_time)
            odd_time_trim = np.copy(odd_time)

            even_idx = idx_arr[:,0,b]
            odd_idx = idx_arr[:,1,b]

            even_idx_trim = even_idx[~even_isnan]
            odd_idx_trim = odd_idx[~odd_isnan]

        while True:        
            madeChange = 0
            if even_time[cap_arr[0,b]-1] > odd_time[0]:
                cap_arr[0,b] -= 1
                madeChange = 1
            if odd_time[cap_arr[1,b]-1] > 40 + even_time[0]:
                cap_arr[1,b] -= 1
                madeChange = 1
            if madeChange == 0:
                break

        if debug:
            even_time_trim = even_time_trim[:cap_arr[0,b]]
            odd_time_trim = odd_time_trim[:cap_arr[1,b]]
            
            even_idx_trim = even_idx_trim[:cap_arr[0,b]]
            odd_idx_trim = odd_idx_trim[:cap_arr[1,b]]

            time_arr_rm_overlap[even_idx_trim.astype(int),0,b] = even_time_trim
            time_arr_rm_overlap[odd_idx_trim.astype(int),1,b] = odd_time_trim
 
            idx_arr_rm_overlap[even_idx_trim.astype(int),0,b] = even_idx_trim
            idx_arr_rm_overlap[odd_idx_trim.astype(int),1,b] = odd_idx_trim

    print(cap_arr)

    print(cap_arr_old - cap_arr)

    print(idx_arr.shape)
    print(time_arr.shape)
    print(cap_arr.shape)
        
    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
            
    # save 
    h5_file_name = dname[:-4]
    h5_file_name += f'.h5'
    hf = h5py.File(Output+h5_file_name, 'w')
    hf.create_dataset('idx_arr', data=idx_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('time_arr', data=time_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('cap_arr', data=cap_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('Vadj_corr_32', data=Vadj_corr_32, compression="gzip", compression_opts=9)
    
    if debug:
        hf.create_dataset('idx_arr_txt', data=idx_arr_txt, compression="gzip", compression_opts=9)
        hf.create_dataset('time_arr_txt', data=time_arr_txt, compression="gzip", compression_opts=9)
        hf.create_dataset('cap_arr_txt', data=cap_arr_txt, compression="gzip", compression_opts=9)
        hf.create_dataset('time_arr_Vadj', data=time_arr_Vadj, compression="gzip", compression_opts=9)
        hf.create_dataset('idx_arr_rm_overlap', data=idx_arr_rm_overlap, compression="gzip", compression_opts=9)
        hf.create_dataset('time_arr_rm_overlap', data=time_arr_rm_overlap, compression="gzip", compression_opts=9)

    hf.close()
    del hf 

    h5_file_name_cap = dname[:-4]
    h5_file_name_cap += f'_CapNum_Only'
    h5_file_name_cap += f'.h5'
    hf = h5py.File(Output+h5_file_name_cap, 'w')
    hf.create_dataset('cap_arr', data=cap_arr, compression="gzip", compression_opts=9)
    hf.close()
    del hf

    print(f'output is {Output}{h5_file_name}')
    file_size = np.round(os.path.getsize(Output+h5_file_name)/1204/1204,2)
    print('file size is', file_size, 'MB')

    print(f'output is {Output}{h5_file_name_cap}')
    file_size_cap = np.round(os.path.getsize(Output+h5_file_name_cap)/1204/1204,2)
    print('file size is', file_size_cap, 'MB')

    print('done!')
 
   
if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 3:
        Usage = """
    This is designed to make sample index table and number of index in each block. Need to choose specific station number.

    If it is data,
    Usage = python3 %s
    <Station ex) 2>
    <Output ex) /home/mkim/analysis/MF_filters/data/>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output = str(sys.argv[2])

    sample_idx_table_maker(CPath = curr_path+'/..', Station = station, Output = output)

del curr_path












    
