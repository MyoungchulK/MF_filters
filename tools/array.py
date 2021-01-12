import numpy as np

def arr_1d(dim0, fill, d_type):

    return np.full((dim0), fill, dtype = d_type)

def arr_2d(dim0, dim1, fill, d_type):

    return np.full((dim0, dim1), fill, dtype = d_type)

def arr_3d(dim0, dim1, dim2, fill, d_type):

    return np.full((dim0, dim1, dim2), fill, dtype = d_type)

def arr_4d(dim0, dim1, dim2, dim3, fill, d_type):

    return np.full((dim0, dim1, dim2, dim3), fill, dtype = d_type)
