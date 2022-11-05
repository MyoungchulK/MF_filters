import os
import numpy as np

def size_checker(d_path, bit_size = False):

    file_size = os.path.getsize(d_path)
    if bit_size == True:
        pass
    else:
        bit_to_mega_byte = 1024**2
        file_size = np.round(file_size/bit_to_mega_byte, 2)
    
    print(f'file size is {file_size} MB')
