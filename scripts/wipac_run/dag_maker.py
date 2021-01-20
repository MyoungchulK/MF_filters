import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm

def data_ped_list(Station, Year):

    if Year == '2013':

        # make data list
        data_list = sorted(glob(f'/data/wipac/ARA/{Year}/filtered/burnSample1in10/ARA0{Station}/root/*/event[0-9][0-9][0-9][0-9].root'))
        data_run = []
        for d in data_list:
            data_run.append(int(d[-9:-5]))
        data_run = np.asarray(data_run)

        # make ped list
        ped_list = sorted(glob(f'/data/user/mkim/ARA_2013_Ped/ARA0{Station}/*pedestalValues*'))
        ped_run = []
        for p in ped_list:
            ped_run.append(int(p[-8:-4]))
        ped_run = np.asarray(ped_run)

        last_ped_list = []
        last_ped_run = np.array([])

    else:

        # make data list
        data_list = sorted(glob(f'/data/wipac/ARA/{Year}/unblinded/L1/ARA0{Station}/*/run*/event[0-9][0-9][0-9][0-9][0-9][0-9].root'))
        data_run = []
        for d in data_list:
            data_run.append(int(d[-9:-5]))
        data_run = np.asarray(data_run)

        # make ped list
        ped_list = sorted(glob(f'/data/wipac/ARA/{Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*'))
        ped_run = []
        for p in ped_list:
            ped_run.append(int(p[-8:-4]))
        ped_run = np.asarray(ped_run)

        if Year == '2014':

            # make last year ped list
            last_Year = str(int(Year)-1)
            last_ped_list = sorted(glob(f'/data/user/mkim/ARA_2013_Ped/ARA0{Station}/*pedestalValues*'))
            last_ped_run = []
            for lp in last_ped_list:
                last_ped_run.append(int(lp[-8:-4]))
            last_ped_run = np.asarray(last_ped_run)

        else:

            # make last year ped list
            last_Year = str(int(Year)-1)
            last_ped_list = sorted(glob(f'/data/wipac/ARA/{last_Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*'))
            last_ped_run = []
            for lp in last_ped_list:
                last_ped_run.append(int(lp[-8:-4]))
            last_ped_run = np.asarray(last_ped_run)

    return data_list, data_run, ped_list, ped_run, last_ped_list, last_ped_run

# argument
Station = str(sys.argv[1])
Year = str(sys.argv[2])
Output = str(sys.argv[3])
DMode = str(sys.argv[4])
print('Station:',Station)
print('Year:',Year)

# data ped list
data_list, data_run, ped_list, ped_run, last_ped_list, last_ped_run = data_ped_list(Station, Year)
print('Data & Ped loading is done!')

# dag info
print('Dag making is starts!')
dag_file_name = f'ARA0{Station}_{Year}.dag'
contents = ""

# set dag path
#dag_path = '/home/mkim/analysis/MF_filters/scripts/wipac_run/'
#if not os.path.exists(dag_path):
#    os.makedirs(dag_path)
#os.chdir(dag_path)

with open(dag_file_name, 'w') as f:
    f.write(contents)

for r in tqdm(range(len(data_run))):

    try:
     
        # check where is the nearest ped run number while it is smaller than data(<0).
        ped_pair_diff = ped_run - data_run[r]
        p_index = np.where(ped_pair_diff<0)[0][-1]

        contents = ""
        contents += f'JOB job_{r} ARA_job.sub \n'
        contents += f'VARS job_{r} data="{data_list[r]}" ped="{ped_list[p_index]}" station="{Station}" run="{data_run[r]}" out="{Output}" mode="{DMode}"\n\n'

        with open(dag_file_name, 'a') as f:
            f.write(contents)

    except IndexError:

        # matching pair with last year pedestral
        try:

            # check where is the nearest ped run number while it is smaller than data(<0).
            ped_pair_diff = last_ped_run - data_run[r]
            p_index = np.where(ped_pair_diff<0)[0][-1]

            print('Correspending pedestal is in last year data pack!')
            print(data_list[r],last_ped_list[p_index])
    
            contents = ""
            contents += f'JOB job_{r} ARA_job.sub \n'
            contents += f'VARS job_{r} data="{data_list[r]}" ped="{last_ped_list[p_index]}" station="{Station}" run="{data_run[r]}" out="{Output}" mode="{DMode}"\n\n'

            with open(dag_file_name, 'a') as f:
                f.write(contents)

        except IndexError:

            if Year == '2013':

                # match with default pedestal
                de_ped = f'/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraRoot/AraEvent/calib/ATRI/araAtriStation{Station}Pedestals.txt' 

                print('Matching data with default pedestal!')
                print(data_list[r], de_ped)

                contents = ""
                contents += f'JOB job_{r} ARA_job.sub \n'
                contents += f'VARS job_{r} data="{data_list[r]}" ped="{de_ped}" station="{Station}" run="{data_run[r]}" out="{Output}" mode="{DMode}"\n\n'

                with open(dag_file_name, 'a') as f:
                    f.write(contents)

            else:

                # trouble run
                print('Weired run! Couldnt find the ped pair')
                print(data_list[r],data_run[r])

print('Dag making is done!')

print('Output is',dag_file_name)

            
