from numba import cuda ,float32
import numpy as np
import time
import torch



@cuda.jit
def coalesced_matmul( a ,b ,c ,block_size ):
  ind1 = cuda.blockIdx.x * block_size + cuda.blockIdx.y * cuda.gridDim.x * block_size
  ind2 = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
  row ,col = ( ind1 + ind2 )//len(c[0]) , (ind1 + ind2 )%len(c[0])
  if row<len(a) and col<len(b[0]) :
    sum = 0
    for i in range(len(a[0])):
      sum += a[row][i] * b[i][col]
    c[row][col] = sum
