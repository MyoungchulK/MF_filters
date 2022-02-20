import numpy as np
import os, sys
import re
from glob import glob
from tqdm import tqdm

def list_maker(glob_path, Station, Year):

    d_list = glob(glob_path)
    d_run_num = []
    for d in d_list:
        run_num = int(re.sub("\D", "", d[-11:]))
        d_run_num.append(run_num)
        del run_num
    d_run_num = np.asarray(d_run_num)

    # sort the run and path
    d_run_idx = np.argsort(d_run_num)
    d_run_num_sort = d_run_num[d_run_idx]
    d_list_sort = []
    for s in range(len(d_run_num_sort)):
        d_list_sort.append(d_list[d_run_idx[s]])
    del d_list, d_run_num, d_run_idx    

    if Station ==3 and Year == 2018:
        digit5_run_idx = np.where(d_run_num_sort >= 10000)[0]
        d_list_sort_2018 = []
        for s in range(len(digit5_run_idx)):
            d_list_sort_2018.append(d_list_sort[digit5_run_idx[s]])
        d_run_num_sort = d_run_num_sort[digit5_run_idx]
        d_list_sort = d_list_sort_2018
    else:
        pass

    return d_list_sort, d_run_num_sort
    

def general_data_ped_list(Station, Year):

    # make data list
    data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/unblinded/L1/ARA0{Station}/[0-9][0-9][0-9][0-9]/run[0-9][0-9][0-9][0-9][0-9][0-9]/event[0-9][0-9][0-9][0-9][0-9][0-9].root', Station, Year)
    #data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/unblinded/L1/ARA0{Station}/*/run*/event[0-9]*.root', Station, Year)

    # make ped list
    ped_list, ped_run = list_maker(f'/data/exp/ARA/{Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, Year)

    # make last year ped list
    last_Year = int(Year-1)
    if last_Year == 2013:
        last_ped_list, last_ped_run = list_maker(f'/data/user/mkim/ARA_{last_Year}_Ped/ARA0{Station}/*pedestalValues*', Station, last_Year)
    elif Year == 2018 and Station == 3:
        last_Year = int(Year-2)
        last_ped_list, last_ped_run = list_maker(f'/data/exp/ARA/{last_Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, last_Year)
    else:
        last_ped_list, last_ped_run = list_maker(f'/data/exp/ARA/{last_Year}/calibration/pedestals/ARA0{Station}/*pedestalValues*', Station, last_Year)

    # total ped list
    ped_list = last_ped_list + ped_list
    ped_run = np.append(last_ped_run, ped_run, axis=0)

    return data_list, data_run, ped_list, ped_run

def A235_data_ped_list(Station, Year):

    if Year == 2013 and Station != 5: #when we were full of hope and dream

        # make data list    
        data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/filtered/burnSample1in10/ARA0{Station}/root/*/event[0-9]*.root', Station, Year)
        
        # total ped list
        ped_list, ped_run = list_maker(f'/data/user/mkim/ARA_2013_Ped/ARA0{Station}/*pedestalValues*', Station, Year)

    elif Year > 2013 and Year < 2017 and Station != 5:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    elif Year == 2017 and Station == 2:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    elif Year == 2018:

        # data & ped list
        data_list, data_run, ped_list, ped_run = general_data_ped_list(Station, Year)

    else:

        print('Wrong Station & Year combination!')
        print('Choose 1) 2013~2016:ARA2&3, 2) 2017:ARA2, 3) 2018:ARA2&3&5')
        sys.exit(1)
        
    return data_list, data_run, ped_list, ped_run

#def dag_statement(r, data_list, ped_list, Output, Station, data_run):
def dag_statement(r, data_list, ped_list, Station, data_run):

    contents = ""
    contents += f'JOB job_{r} ARA_job.sub \n'
    #contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" out="{Output}" station="{Station}" run="{data_run}"\n\n'
    contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" station="{Station}" run="{data_run}"\n\n'

    return contents

# argument
Station = int(sys.argv[1])
Year = int(sys.argv[2])
#Output = str(sys.argv[3])
if Station != 2 or Station != 3 or Station != 5:
    if Year < 2013 or Year > 2018 :
        print('Wrong Station & Year combination!')
        print('Choose 1) 2013~2016:ARA2&3, 2) 2017:ARA2, 3) 2018:ARA2&3&5')
        sys.exit(1)

print('Station:',Station)
print('Year:',Year)

# data ped list
print('Loading Data & Ped ...')
data_list, data_run, ped_list, ped_run = A235_data_ped_list(Station, Year)
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

# dag contents
with open(dag_file_name, 'w') as f:
    f.write(contents)

for r in tqdm(range(len(data_run))):

    try:
     
        # check where is the nearest ped run number while it is smaller than data(<0).
        ped_pair_diff = ped_run - data_run[r]
        p_index = np.where(ped_pair_diff<0)[0][-1]

        # write contents
        #contents = dag_statement(r, data_list[r], ped_list[p_index], Output, Station, data_run[r])
        contents = dag_statement(r, data_list[r], ped_list[p_index], Station, data_run[r])
        with open(dag_file_name, 'a') as f:
            f.write(contents)

    except IndexError:

        if Year == 2013 or Station == 5:# or (Year == 2018  and Station  == 3):

            # match with default pedestal
            de_ped = f'/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraRoot/AraEvent/calib/ATRI/araAtriStation{Station}Pedestals.txt' 

            # write contents
            #contents = dag_statement(r, data_list[r], de_ped, Output, Station, data_run[r])
            contents = dag_statement(r, data_list[r], de_ped, Station, data_run[r])
            with open(dag_file_name, 'a') as f:
                f.write(contents)

            print('Matching data with default pedestal!')
            print(data_list[r], de_ped)

        else:

            # trouble run
            print('Weired run! Couldnt find the ped pair')
            print(data_list[r],data_run[r])

print('Dag making is done!')

print('Output is',dag_file_name)

            
