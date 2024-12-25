from numba import cuda ,float32
import numpy as np
import time
import torch

@cuda.jit
def strided_access( a ):
  ind = cuda.threadIdx.x
  if ind < len(a) :
    curr_val = a[ind][0]
    curr_val = curr_val + ind
    a[ind][0] = curr_val

@cuda.jit
def continuous_access(a):
  ind = cuda.threadIdx.x
  if ind < len(a[0]) :
    curr_val = a[0][ind]
    curr_val = curr_val + ind
    a[0][ind] = curr_val