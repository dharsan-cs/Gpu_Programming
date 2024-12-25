from numba import cuda ,float32
import numpy as np
import time
import torch


@cuda.jit
def tile_matmul( a ,b ,c ,n ):
  shared_a = cuda.shared.array((16 ,16) , dtype = float32)
  shared_b = cuda.shared.array((16 ,16) , dtype = float32)

  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y
  row = cuda.blockIdx.y * 16 + ty
  col = cuda.blockIdx.x * 16 + tx

  if row<len(a) and col<len(c[0]) :
    result = 0

    for tile_id in range( 0 ,n//16):

      shared_a[ty][tx] = a[row][ tile_id * 16 + tx ]
      shared_b[ty][tx] = b[ tile_id * 16 + ty ][col]
      cuda.syncthreads( )

      for i in range(16) :
        result = shared_a[ty][i] * shared_b[i][tx]
      cuda.syncthreads( )

    c[row][col] = result