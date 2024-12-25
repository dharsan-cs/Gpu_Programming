from numba import cuda ,float32
import numpy as np
import time
import torch


@cuda.jit
def normal_matmul( a ,b ,c ):
  x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
  if x<len(a) and y<len(b[0]):
    sum = 0
    for i in range(len(a[0])):
      sum += a[x][i] * b[i][y]
    c[x][y] = sum