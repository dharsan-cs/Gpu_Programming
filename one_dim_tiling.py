from numba import cuda ,float32
import numpy as np
import time
import torch


@cuda.jit
def one_dim_block_tiling( a ,b ,c ):
  assert bm*bk ==  cuda.blockDim.x ,"block size is not equal to bm*bk"
  assert bn*bk ==  cuda.blockDim.x ,"block size is not equal to bn*bk"
  assert cuda.blockDim.x == (bn*bm)//tm ,"number of threads is not correct"

  a_s = cuda.shared.array( (bm ,bk) ,dtype = float32 ) #shared memory in l1 cache for storing A mat tiles
  b_s = cuda.shared.array( (bk ,bn) ,dtype = float32 ) #shared memory in l1 cache for storing B mat tiles

  row = cuda.blockIdx.y * bm  #starting row in mat A
  col = cuda.blockIdx.x * bn  #starting col in mat B

  #get row ,col value corresponding to shared A mat and shared B mat
  sa_row ,sa_col = cuda.threadIdx.x//bk ,cuda.threadIdx.x%bk
  sb_row ,sb_col = cuda.threadIdx.x//bn ,cuda.threadIdx.x%bn

  thread_row = cuda.threadIdx.x//bn
  thread_col = cuda.threadIdx.x%bn

  # Local array to store results
  results = cuda.local.array( shape = (tm,) ,dtype = float32 )
  for i in range(tm):
    results[i] = 0.0

  for tile_id in range(0 ,k//bk ):
    #loading the data to the shared memory
    a_s[ sa_row ][ sa_col ] = a[ row + sa_row ][ tile_id * bk + sa_col ]
    b_s[ sb_row ][ sb_col ] = b[ tile_id * bk + sb_row ][ col + sb_col ]
    cuda.syncthreads( )

    ##performing computation for tm rows in a matrix
    for dotidx in range(bk):
      tmp = b_s[ dotidx ][ thread_col ]
      for i in range( tm ) :
        results[i] = results[i] + a_s[ thread_row * tm + i][ dotidx ]
    cuda.syncthreads( )

  ## assigning the output values
  for i in range(tm):
    c[ row + thread_row*tm + i][ col + thread_col ] = results[i]