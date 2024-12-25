from numba import cuda  
import numpy as np 
import time


"""
Note :- 
-> using streams didnot provide a significant improment in time 
-> but data_processing - data_movement parallelism is noticible
-> reason could time( data_processing ) << time( data_movement )
"""

""" 
This class given set matrix [(mata1 ,matb1 ,matc1) ,(mata2 ,matb2 ,matc2) .... ] ,it perform matrix multiplication process on
all the datas in dataset efficiently using cuda streams
"""

class matrix_multiplication:

  def __init__(self ):
    self.data_transfer_stream = cuda.stream( )
    self.data_process_stream = cuda.stream( )
    self.sync_event = cuda.event()
   
  def process_data_set(self ,data_set ,ndim ):
    result = [ ]

    ## host-to-device (loading all the data from host to device sequentialy)
    device_dataset = [ ]
    for h_a ,h_b ,h_c in data_set: 
      d_a = cuda.to_device(h_a)
      d_b = cuda.to_device(h_b)
      d_c = cuda.to_device(h_c)
      device_dataset.append((d_a ,d_b ,d_c))
    
    ##parallelising device-to-host memory transfer and data processing 
    grid ,block = self.__grid_block( n_dim = ndim )
    
    ##performing two-way concurrency
    normal_matmul[grid ,block ,self.data_process_stream ](device_dataset[0][0] ,device_dataset[0][1] ,device_dataset[0][2])    
    for i in range( 1 ,len(data_set) ):
      self.data_process_stream.synchronize()
      normal_matmul[grid ,block ,self.data_process_stream ](device_dataset[i][0] ,device_dataset[i][1] ,device_dataset[i][2])      
      result.append(device_dataset[i-1][2].copy_to_host( stream = self.data_transfer_stream) )
    self.data_process_stream.synchronize()
    result.append(device_dataset[i-1][2].copy_to_host( stream = self.data_transfer_stream) )

    #making host wait 
    cuda.synchronize()

    return result
    
  def __grid_block(self ,n_dim):
    assert n_dim > 32 ,'matrix size is less than 32'
    block = (32 ,32 ,1 )
    grid_x = (n_dim + block[0] - 1) // block[0]
    grid_y = (n_dim + block[1] - 1) // block[1]
    grid = (grid_x ,grid_y ,1)
    return grid ,block