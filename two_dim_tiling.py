from numba import cuda ,float32
import numpy as np
import time
import torch


@cuda.jit
def two_dim_block_tiling( a ,b ,c ):
  assert cuda.blockDim.x == (bm*bn)//(tm*tn) ,"number of threads is not correct"
  #declaring the shared memory
  s_a = cuda.shared.array( (bm,bk) ,dtype = float32 )
  s_b = cuda.shared.array( (bk,bn) ,dtype = float32)

  #starting row of A matrix and starting column of B matrix
  row = cuda.blockIdx.y * bm
  col = cuda.blockIdx.x * bn

  #get row ,col value corresponding to shared A mat and shared B mat
  sa_row ,sa_col ,stride_a = cuda.threadIdx.x//bk ,cuda.threadIdx.x%bk ,cuda.blockDim.x//bk
  sb_row ,sb_col ,stride_b = cuda.threadIdx.x//bn ,cuda.threadIdx.x%bn ,cuda.blockDim.x//bn

  #thread row and thread col
  thread_row = cuda.threadIdx.x // (bn//tn)
  thread_col = cuda.threadIdx.x % (bn//tn)

  ## declaring variables to store the results
  result_a = cuda.local.array( shape = (tm,) ,dtype = float32)
  result_b = cuda.local.array( shape = (tn,) ,dtype = float32)
  result = cuda.local.array( shape = (tm,tn) ,dtype = float32)
  for i in range(tm):
    for j in range(tn):
      result[i][j] = 0

  for tile_id in range( 0 ,k//bk ):
    ## loading the data from A matrix
    for i in range( sa_row ,bm ,stride_a ):
      s_a[i][sa_col] = a[row + i][ tile_id*bk + sa_col]

    ## loading the data from B matrix
    for i in range( sb_row ,bk ,stride_b ):
      s_b[i][sb_col] = b[ tile_id*bk + i ][col + sb_col]
    cuda.syncthreads()

    ##performing computation for tm x tn matrix corresponding to thread
    for dotidx in range(bk):
      for i in range(tm):
        result_a[i] = s_a[thread_row*tm+i][dotidx]
      for i in range(tn) :
        result_b[i] = s_b[dotidx][thread_col*tn+i]
      for i in range(tm):
        for j in range(tn):
          result[i][j] = result[i][j] + result_a[i]*result_b[j]
    cuda.syncthreads()

  #storing the results in c matrix
  for i in range(tm):
    for j in range(tn):
      c[row + thread_row*tm + i][col + thread_col*tn + j] = result[i][j]
