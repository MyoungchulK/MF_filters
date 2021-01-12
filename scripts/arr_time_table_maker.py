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
del curr_path

def arr_table_maker(Station, Grid_Size, Search_Len, Output):

    # load ara root lib
    ROOT = ara_root_lib()

    # number of the antenna
    num_Antennas = antenna_info()[2]

    # cartesian coord of the target
    trg_xyz = ant_xyz(ROOT, Station, num_Antennas)
    del ROOT

    # arrival time delay by plane wave front
    path_dT, path_dT_avg, arr_max_len, nadir_range, phi_range = plane_table(num_Antennas, trg_xyz.T, 1.786
                                                       , nadir_min = 0 + Grid_Size/2, nadir_max = 180
                                                       , phi_min = 0 + Grid_Size/2, phi_max = 360
                                                       , angle_width = Grid_Size)
    del num_Antennas

    # index selection by arrival time delay    
    mov_index, pad_t, pad_len_front, pad_len_end, ps_len_index, mov_t = mov_index_table(path_dT_avg, arr_max_len, Search_Len)

    # creat output file
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name='Plane_Table_A'+str(Station)+'_GS'+str(Grid_Size)+'_PW'+str(Search_Len)+'.h5'
    hf = h5py.File(h5_file_name, 'w')
    del h5_file_name

    # saving station info
    g0 = hf.create_group('Station')
    g0.create_dataset('XYZ(m)', data=trg_xyz, compression="gzip", compression_opts=9)
    g0.create_dataset('St_Num', data=np.array([Station]), compression="gzip", compression_opts=9)
    del g0, trg_xyz

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

    hf.close()
    del hf

    # save only necessary info for actual analysis
    h5_file_name='Plane_Table_A'+str(Station)+'_GS'+str(Grid_Size)+'_PW'+str(Search_Len)+'_lite.h5'
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
    if len (sys.argv) !=5:
        Usage = """
    It will generate arrival time table by AraSim in ARA cvmfs.
    Table variables are 1) Theta, 2) Phi, 3) Radius, 4) antenna and 5) ray solution
    Output table dimesion would be (theta, phi, radius, antenna, ray solution).

    Usage = python3 %s
    <Station ex)2>
    <Grid size ex)36>
    <Search lengh ex)50>
    <Output path ex)/data/user/mkim/OMF_sky/Table/>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Station=int(sys.argv[1])
    Grid_Size=int(sys.argv[2])
    Search_Len=int(sys.argv[3])
    Output=str(sys.argv[4])

    arr_table_maker(Station, Grid_Size, Search_Len, Output)



























