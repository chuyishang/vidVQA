from numba import cuda
cuda.select_device(2)
cuda.close()