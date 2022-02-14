import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_root import ara_root_lib
from tools.ara_root import ant_xyz
from tools.antenna import antenna_info
from tools.arr_table import plane_table 
from tools.arr_table import mov_index_table 
from tools.arr_table import nz_ice 
del curr_path

def arr_table_maker(Station, Years, Grid_Size, Search_Len, Output):

    # load ara root lib
    ROOT = ara_root_lib()

    # number of the antenna
    num_Antennas = antenna_info()[2]

    # cartesian coord of the target
    trg_xyz = ant_xyz(ROOT, Station, num_Antennas, Years)
    trg_xyz_T = np.copy(trg_xyz.T) 
    del ROOT

    # ice model. From Kaeli
    A = 1.78
    B = 1.326
    C = 0.0202
    nz = np.nanmean(nz_ice(trg_xyz_T[2], A, B, C))

    # arrival time delay by plane wave front
    path_dT, path_dT_avg, arr_max_len, nadir_range, phi_range = plane_table(num_Antennas, trg_xyz_T, nz
                                                       , nadir_min = 0 + Grid_Size/2, nadir_max = 180
                                                       , phi_min = 0 + Grid_Size/2, phi_max = 360
                                                       , angle_width = Grid_Size)
    del num_Antennas, trg_xyz_T

    # index selection by arrival time delay    
    mov_index, pad_t, pad_len_front, pad_len_end, ps_len_index, mov_t = mov_index_table(path_dT_avg, arr_max_len, Search_Len)

    # creat output file
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Plane_Table_A{Station}_Y{Years}_GS{Grid_Size}_PW{Search_Len}.h5'
    hf = h5py.File(h5_file_name, 'w')
    del h5_file_name

    # saving station info
    g0 = hf.create_group('Station')
    g0.create_dataset('XYZ(m)', data=trg_xyz, compression="gzip", compression_opts=9)
    g0.create_dataset('St_Num', data=np.array([Station]), compression="gzip", compression_opts=9)
    del g0

    # saving arrival time info
    g1 = hf.create_group('Arr_Table')
    g1.create_dataset('path_dT(ns)', data=path_dT, compression="gzip", compression_opts=9)
    g1.create_dataset('path_dT_avg(ns)', data=path_dT_avg, compression="gzip", compression_opts=9)
    g1.create_dataset('path_dT_max(ns)', data=np.array([arr_max_len]), compression="gzip", compression_opts=9)
    g1.create_dataset('Nadir(deg)', data=nadir_range, compression="gzip", compression_opts=9)
    g1.create_dataset('Phi(deg)', data=phi_range, compression="gzip", compression_opts=9)
    g1.create_dataset('Grid_Size(deg)', data=np.array([Grid_Size]), compression="gzip", compression_opts=9)
    del g1, path_dT, path_dT_avg, arr_max_len, nadir_range, phi_range       
 
    #saving selected index info
    g2 = hf.create_group('Index_Table')
    g2.create_dataset('mov_index', data=mov_index, compression="gzip", compression_opts=9)
    g2.create_dataset('mov_t', data=mov_t, compression="gzip", compression_opts=9)
    g2.create_dataset('pad_t', data=pad_t, compression="gzip", compression_opts=9)
    g2.create_dataset('pad_len_front', data=np.array([pad_len_front]), compression="gzip", compression_opts=9)
    g2.create_dataset('pad_len_end', data=np.array([pad_len_end]), compression="gzip", compression_opts=9)
    g2.create_dataset('ps_len_index', data=np.array([ps_len_index]), compression="gzip", compression_opts=9)
    g2.create_dataset('Search_Len(ns)', data=np.array([Search_Len]), compression="gzip", compression_opts=9)
    del g2

    # saving ice model info
    g3 = hf.create_group('Ice_Model')
    g3.create_dataset('nz', data=np.array([nz]), compression="gzip", compression_opts=9)
    g3.create_dataset('A', data=np.array([A]), compression="gzip", compression_opts=9)
    g3.create_dataset('B', data=np.array([B]), compression="gzip", compression_opts=9)
    g3.create_dataset('C', data=np.array([C]), compression="gzip", compression_opts=9)
    del g3, nz, A, B, C

    hf.close()
    del hf

    # save only necessary info for actual analysis
    h5_file_name=f'Ant_Pos_A{Station}_Y{Years}.h5'
    hf = h5py.File(h5_file_name, 'w')
    del h5_file_name

    # saving arrival time info
    hf.create_dataset('Ant_Pos', data=trg_xyz, compression="gzip", compression_opts=9)
    del trg_xyz

    hf.close()
    del hf

    # save only necessary info for actual analysis
    h5_file_name=f'Plane_Table_A{Station}_Y{Years}_GS{Grid_Size}_PW{Search_Len}_lite.h5'    
    hf = h5py.File(h5_file_name, 'w')
    del Station, Grid_Size, Search_Len

    #saving selected index info
    hf.create_dataset('mov_index', data=mov_index, compression="gzip", compression_opts=9)
    hf.create_dataset('mov_t', data=mov_t, compression="gzip", compression_opts=9)
    hf.create_dataset('pad_t', data=pad_t, compression="gzip", compression_opts=9)
    hf.create_dataset('pad_len_front', data=np.array([pad_len_front]), compression="gzip", compression_opts=9)
    hf.create_dataset('pad_len_end', data=np.array([pad_len_end]), compression="gzip", compression_opts=9)
    hf.create_dataset('ps_len_index', data=np.array([ps_len_index]), compression="gzip", compression_opts=9)
    del mov_index, pad_t, pad_len_front, pad_len_end, ps_len_index, mov_t

    hf.close()
    del hf
    
    print('output is',Output+h5_file_name)
    del Output, h5_file_name
    print('done!')
    
if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=6:
        Usage = """
    It will generate giant index array based on plane wavefront concept of table by AraSim in ARA cvmfs.
    Output table will be contained 1)index array, 2)shift array, 3)pad array, 4)begining and end of pad array
    , and 5)peak search width.

    Usage = python3 %s
    <Station ex)2>
    <Years ex)2014>
    <Grid size ex)36>
    <Search lengh ex)50>
    <Output path ex)/data/user/mkim/OMF_sky/Table/>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Station=int(sys.argv[1])
    Years=int(sys.argv[2])
    Grid_Size=int(sys.argv[3])
    Search_Len=int(sys.argv[4])
    Output=str(sys.argv[5])

    arr_table_maker(Station, Years, Grid_Size, Search_Len, Output)



























