import os
import numpy as np

def size_checker(d_path, use_byte = False, use_print = False):

    dat_size = ['Bytes', 'KB', 'MB', 'GB', 'TB!?!?!', 'PB!?!?!']
    dat_count = 0

    file_size = os.path.getsize(d_path)
    if use_byte:
        pass
    else:
        n = len(str(file_size).split(".")[0])
        while n > 3:
            file_size /= 1024
            n = len(str(file_size).split(".")[0])
            dat_count += 1
        file_size = np.round(file_size, 2)
    
    msg = f'file size is {file_size} {dat_size[dat_count]}'
    if use_print:
        print(msg)

    return msg
