import os, sys
from glob import glob
from subprocess import call
from tqdm import tqdm

def ped_unzip_2013(Station, Output):
    # tar file list in cobalt
    ped_raw_list = sorted(glob(f'/data/exp/ARA/2013/filtered/ARA0{Station}-pedestals/*/SPS-ARA-PEDESTAL-run-[0-9][0-9][0-9][0-9][0-9][0-9].ARA0{Station}.tar.gz'))

    # create output path
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)

    # create temp path
    temp_path = f'{Output}ARA0{Station}/temp/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # 1st untar raw to temp path
    print('1st untar')
    for r in tqdm(ped_raw_list):

        TAR_CMD=f'tar -xf {r} -C {temp_path}'
        call(TAR_CMD.split(' '))

    print('done!')

    # 2nd untar dat file
    ped_dat_list = sorted(glob(f'{temp_path}SPS-ARA-PEDESTAL-run-[0-9][0-9][0-9][0-9][0-9][0-9].ARA0{Station}.dat.tar'))

    print('2nd untar')
    for d in tqdm(ped_dat_list):

        TAR_CMD=f'tar -xf {d} -C {temp_path}'
        call(TAR_CMD.split(' '))

    print('done!')

    # move pedestals
    ped_file_list = sorted(glob(f'{temp_path}ARA0{Station}/2013/raw_data/run_[0-9][0-9][0-9][0-9][0-9][0-9]/*'))

    # create ped path
    ped_path = f'{Output}ARA0{Station}/'
    if not os.path.exists(ped_path):
        os.makedirs(ped_path)

    print('move ped')
    for f in tqdm(ped_file_list):

        MV_CMD=f'mv {f} {ped_path}'
        call(MV_CMD.split(' '))

    print('done!')

    print('remove temp dir')

    RM_CMD = f'rm -rf {temp_path}'
    call(RM_CMD.split(' '))

    print('done!')

    print('output is in',ped_path)

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 3:
        Usage = """

    Usage = python3 %s <Station ex)2> <Output path:/home/mkim/>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #qrgv
    Station = str(sys.argv[1])
    Output = str(sys.argv[2])
    
    ped_unzip_2013(Station, Output)













