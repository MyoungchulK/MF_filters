import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_data_load import ara_geom_loader

def ped_arr(ped_reped):

    # config
    useful_ch = 4
    nblk = 512
    samp_per_block = 64
    num_ant = 16
   
    # re-align to rf ch order
    ped_reped_arr = np.full((nblk*samp_per_block, 32),np.nan)
    for s in range(ped_reped.shape[0]):
            dda_c = int(ped_reped[s,2])
            dda_b = int(ped_reped[s,0])
            blk = int(ped_reped[s,1])
            samp = ped_reped[s,3:]
            rf_ch = dda_b + dda_c * 4
            ped_reped_arr[blk*samp_per_block:blk*samp_per_block+samp_per_block,rf_ch] = samp
    
    print(ped_reped_arr.shape)   
    ara_geom = ara_geom_loader(3, 2013)
    ele_ch = ara_geom.get_ele_ch_idx()
    ped_reped_arr = ped_reped_arr[:, ele_ch]
    print(ped_reped_arr.shape)

    return ped_reped_arr

def ped_h5_converter(Station, Output, Ped_type):

    if Station == 2 or Station == 3:
        pass
    else:
        print('It is only working for A2/3, Need to modified ped_arr for different station!')
        sys.exit(1)

    # make ped list
    if Ped_type == 'repeder':
        p_path = glob(f'/data/user/mkim/OMF_filter/ARA0{Station}/Ped/pedestalValues.run*')
    elif Ped_type == 'closest':
        p_path = glob(f'/data/user/mkim/ARA_2013_Ped/ARA0{Station}/pedestalValues.run*')
        for year in range(2014,2019):
            if year == 2017 and Station ==3:
                pass
            p_path += glob(f'/data/exp/ARA/{year}/calibration/pedestals/ARA0{Station}/pedestalValues.run*')
    elif Ped_type == 'built_in':
        p_path = glob(f'/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraRoot/AraEvent/calib/ATRI/araAtriStation{Station}Pedestals.txt')
    else:
        print('Wrong Ped. type!')
        sys.exit(1)

    print('Total # of Ped:',len(p_path))

    # create output path
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)

    #log bad ped
    bad_p_name = f'A{Station}_Bad_Ped.txt'
    with open(Output + bad_p_name, 'w') as f:
        f.write("")
    bad_p_count = 0

    for p_file in tqdm(p_path):
      if p_file[-9:-4] == 'n1804':   

        # load ped
        try:
            ped = np.loadtxt(p_file)
        except ValueError:
            print(f'Bad (Value) Ped.!, {p_file}')
            with open(Output + bad_p_name, 'a') as f:
                f.write(f'{p_file} \n')
            bad_p_count += 1
            continue       

        # checking 
        p_shape = ped.shape
        if p_shape[0] == 16384 and p_shape[1] == 67:
            pass
        else:
            print(f'Bad Ped.!, {p_file}, Shape:{p_shape}')
            with open(Output + bad_p_name, 'a') as f:
                f.write(f'{p_file} \n')
            bad_p_count += 1
            continue
       
        # convert
        ped_rf_arr = ped_arr(ped)
        
        # save as h5
        anchor_idx = p_file[::-1].find('/')
        h5_file_name = p_file[-anchor_idx:-4]
        h5_file_name += f'.h5'
        hf = h5py.File(Output + h5_file_name, 'w')
        hf.create_dataset('ped_rf_arr', data=ped_rf_arr, compression="gzip", compression_opts=9)
        hf.close()
        del hf, h5_file_name, ped, anchor_idx

    with open(Output + bad_p_name, 'a') as f:
        f.write(f'Number of bad Ped.: {bad_p_count} \n')
        f.write(f'Number of good Ped.: {len(p_path) - bad_p_count} \n')
        f.write(f'Number of total Ped.: {len(p_path)} \n')
    print('done!')
    print('Number of bad Ped.:',bad_p_count)
    print('output is in',Output)

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 4:
        Usage = """

    Usage = python3 %s <Station ex)2> <Output path:/home/mkim/> <Ped type ex) repeder, closest or built_in>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #qrgv
    Station = int(sys.argv[1])
    Output = str(sys.argv[2])
    Ped_type = str(sys.argv[3])

    ped_h5_converter(Station, Output, Ped_type)




















