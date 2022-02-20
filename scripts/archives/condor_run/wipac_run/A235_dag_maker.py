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
        wrong_digit_run_idx = np.where(d_run_num_sort < 10000)[0]
        digit_run_num = d_run_num_sort[wrong_digit_run_idx]
        print('Wrong Run#!:',digit_run_num)

        digit5_run_idx = np.where(d_run_num_sort >= 10000)[0]
        d_list_sort_2018 = []
        for s in range(len(digit5_run_idx)):
            d_list_sort_2018.append(d_list_sort[digit5_run_idx[s]])
        d_run_num_sort = d_run_num_sort[digit5_run_idx]
        d_list_sort = d_list_sort_2018
    else:
        wrong_digit_run_idx = np.where(d_run_num_sort < 100)[0]
        digit_run_num = d_run_num_sort[wrong_digit_run_idx]
        print('Wrong Run#!:',digit_run_num)

        digit3_run_idx = np.where(d_run_num_sort >= 100)[0]
        d_list_sort_digit3 = []
        for s in range(len(digit3_run_idx)):
            d_list_sort_digit3.append(d_list_sort[digit3_run_idx[s]])
        d_run_num_sort = d_run_num_sort[digit3_run_idx]
        d_list_sort = d_list_sort_digit3

    return d_list_sort, d_run_num_sort

def repeder_list_maker(d_path, data_run):

    ped_list = []
    # need to make sure there is ped file in the path and not to AraRoot to use default ped automatically when there is no ped file...
    temp_ped_list = os.listdir(d_path)
    no_ped_list = []
    for p in range(len(data_run)):
        ped_file = f'pedestalValues.run{data_run[p]}.dat'
        if ped_file in temp_ped_list:
            ped_list.append(d_path+ped_file)
        else:
            ped_list.append('None!!!!')
            no_ped_list.append(data_run[p])

    no_ped_list = np.asarray(no_ped_list)
    if len(no_ped_list) != 0:
        print(f'Total {len(no_ped_list)} runs have a no Ped. Run#:',no_ped_list)
        print('Run the repeder again!')
        sys.exit(1)
    else:
        print('All Data has a perfect Ped!')        

    return ped_list

def dag_statement(r, data_list, ped_list, Station, data_run):

    contents = ""
    contents += f'JOB job_{r} ARA_job.sub \n'
    contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" station="{Station}" run="{data_run}"\n\n'

    return contents

# argument
Station = int(sys.argv[1])
Year = int(sys.argv[2])
#Output = str(sys.argv[3])
if Station == 2 and (Year > 2012 and Year < 2019):
    pass
elif (Station == 3 and (Year > 2012 and Year < 2017)) or (Station == 3 and Year == 2018):
    pass
elif Station == 5 and Year == 2018:
    pass
else:
    print('Wrong Station & Year combination!')
    print('Choose 1) 2013~2016:ARA2&3, 2) 2017:ARA2, 3) 2018:ARA2&3&5')
    sys.exit(1)

print('Station:',Station)
print('Year:',Year)

# set dag path
#dag_path = '/home/mkim/analysis/MF_filters/scripts/wipac_run/'
#if not os.path.exists(dag_path):
#    os.makedirs(dag_path)
#os.chdir(dag_path)

# data ped list
print('Loading Data & Ped ...')
if Year == 2013:
    data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/filtered/burnSample1in10/ARA0{Station}/root/*/event[0-9]*.root', Station, Year)
else:
    data_list, data_run = list_maker(f'/data/exp/ARA/{Year}/unblinded/L1/ARA0{Station}/[0-9][0-9][0-9][0-9]/run[0-9][0-9][0-9][0-9][0-9][0-9]/event[0-9][0-9][0-9][0-9][0-9][0-9].root', Station, Year)
ped_list = repeder_list_maker(f'/data/user/mkim/OMF_filter/ARA0{Station}/Ped/',data_run)
print('Data & Ped loading is done!')

# dag info
print('Dag making is starts!')
dag_file_name = f'ARA0{Station}_{Year}.dag'
contents = ""

# dag contents
with open(dag_file_name, 'w') as f:
    f.write(contents)

for r in tqdm(range(len(data_run))):

    # write contents
    contents = dag_statement(r, data_list[r], ped_list[r], Station, data_run[r])
    with open(dag_file_name, 'a') as f:
        f.write(contents)

print('Dag making is done!')

print('Output is',dag_file_name)

            
