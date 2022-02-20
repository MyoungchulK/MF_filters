import numpy as np
import os, sys
import re
from glob import glob
from tqdm import tqdm

def file_sorter(d_path):

    # data path
    d_list_chaos = glob(d_path)
    d_len = len(d_list_chaos)
    print('Total Runs:',d_len)

    # make run list
    run_tot=np.full((d_len),-1,dtype=int)
    aa = 0
    for d in d_list_chaos:
        run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
        aa += 1
    del aa

    # sort the run and path
    run_index = np.argsort(run_tot)
    run_tot = run_tot[run_index]
    d_list = []
    for d in range(d_len):
        d_list.append(d_list_chaos[run_index[d]])
    del d_list_chaos, d_len, run_index

    run_range = np.arange(run_tot[0],run_tot[-1]+1)

    return d_list, run_tot, run_range

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
    ped_run = []
    for p in range(len(data_run)):
        ped_file = f'pedestalValues.run{data_run[p]}.dat'
        if ped_file in temp_ped_list:
            ped_list.append(d_path+ped_file)
            ped_run.append(data_run[p])
        else:
            print('None!!!!')
            ped_list.append('None!!!!')
            no_ped_list.append(data_run[p])

    ped_run = np.asarray(ped_run)
    no_ped_list = np.asarray(no_ped_list)
    if len(no_ped_list) != 0:
        print(f'Total {len(no_ped_list)} runs have a no Ped. Run#:',no_ped_list)
        print('Run the repeder again!')
        sys.exit(1)
    else:
        print('All Data has a perfect Ped!')

    return ped_list, ped_run

Station = int(sys.argv[1])

# sort
n_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Mean_Blk/*'
n_list, n_run_tot, n_run_range = file_sorter(n_path)
del n_run_range

data_list, data_run = list_maker(f'/data/exp/ARA/2013/filtered/burnSample1in10/ARA0{Station}/root/*/event[0-9]*.root', Station, 2013)

for y in range(2014,2019):

    if Station == 3 and y == 2017:
        continue

    data_list_y, data_run_y = list_maker(f'/data/exp/ARA/{y}/unblinded/L1/ARA0{Station}/[0-9][0-9][0-9][0-9]/run[0-9][0-9][0-9][0-9][0-9][0-9]/event[0-9][0-9][0-9][0-9][0-9][0-9].root', Station, y)

    data_list += data_list_y
    data_run = np.append(data_run, data_run_y)

ped_list, ped_run = repeder_list_maker(f'/data/user/mkim/OMF_filter/ARA0{Station}/Ped/', data_run)
print('Data & Ped loading is done!')
print(len(data_run))

def dag_statement(r, data_list, ped_list, Station, data_run):

    contents = ""
    contents += f'JOB job_{r} ARA_job.sub \n'
    contents += f'VARS job_{r} data="{data_list}" ped="{ped_list}" station="{Station}" run="{data_run}"\n\n'

    return contents

# dag info
print('Dag making is starts!')
dag_file_name = f'ARA0{Station}_recovr.dag'
contents = ""

# dag contents
#with open(dag_file_name, 'w') as f:
#    f.write(contents)

for r in tqdm(range(len(data_run))):

    if data_run[r] not in n_run_tot:

        run_idx = np.where(data_run == data_run[r])[0]
        if len(run_idx) > 0:
            run_list = data_list[run_idx[0]]
        else:
            print('There is no run!')
            print(data_run[r])
            sys.exit(1)

        ped_idx = np.where(ped_run == data_run[r])[0]
        if len(ped_idx) > 0:
            n_ped_list = ped_list[ped_idx[0]]
        else:
            print('There is no ped!')
            print(data_run[r])
            sys.exit(1)
 
        print(data_run[r])
        print(run_list)
        print(n_ped_list)
        # write contents
        contents = dag_statement(r, run_list, n_ped_list, Station, data_run[r])
        with open(dag_file_name, 'a') as f:
            f.write(contents)

print('Dag making is done!')

print('Output is',dag_file_name)




















