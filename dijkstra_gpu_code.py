from numba import cuda ,float32
import numpy as np
import numba
import time


##gpu code for running dijkstra 
"""
Note : this code is is not working correctly 100% which means : 
1. if execute a same code multiple times with multiple inputs ,one after the other ,code gives
cuda unknown error at some point .

"""
@cuda.jit
def djikstra_kernel( dist ,par ,visited ,distance_mat):
  assert cuda.threadIdx.x*batch_size < len(dist) ,"Thread.idx is invalid"

  ## finding the ending
  str_ind = cuda.threadIdx.x * batch_size
  end = min( len(dist) ,str_ind + batch_size )
  shared_dist = cuda.shared.array( (block_size,) ,dtype = numba.float64 )
  shared_index = cuda.shared.array( (block_size,)  ,dtype = numba.int32 )

  for _ in range( len(dist) ):

    ##finding non-visted node with minimum distance within this batch
    batch_min_ind = -1
    batch_min_dist = float('inf')
    for i in range(str_ind ,end):
      if visited[i] == 0 and dist[i] < batch_min_dist :
        batch_min_dist = dist[i]
        batch_min_ind = i

    ##updating the found node into dist_array ,index_array in shared memory
    shared_dist[ cuda.threadIdx.x ] = batch_min_dist
    shared_index[ cuda.threadIdx.x ] = batch_min_ind

    ##making sure that all the threads as updated the batch_min_dist ,batch_min_ind
    cuda.syncthreads()

    ##from the updated common array finding globel_min_dist ,globel_min_index
    globel_min_ind = -1
    globel_min_dist = float('inf')
    for i in range( block_size ):
      if shared_dist[i] < globel_min_dist:
        globel_min_dist = shared_dist[i]
        globel_min_ind = shared_index[i]

    ##making sure that every thread has found a globel minimum node
    cuda.syncthreads()

    ##cheacking whether choosen node comes within this batch
    if globel_min_ind >= str_ind and globel_min_ind < end :
      visited[globel_min_ind] = 1

    ##performing the relaxing operation
    for i in range(str_ind ,end):
      weight = distance_mat[globel_min_ind][i]
      if visited[i] == 0 and weight > 0:
        new_dist = globel_min_dist + weight
        if dist[i] > new_dist:
          dist[i] = new_dist
          par[i] = globel_min_ind

    ##make sure all the threads has completed relaxation part
    cuda.syncthreads()

"""
num_node = 3200
speed up (cpu_time/gpu_time): 29.280073451546762
dist accuarcy : 1.0
parent accuracy : 1.0
"""
