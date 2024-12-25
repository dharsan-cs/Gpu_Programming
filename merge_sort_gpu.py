from numba import cuda 
import numpy as np 


@cuda.jit
def sort_kernel( arr ,buffer ,m_size ):
  ind = cuda.threadIdx.x 
  str_pos = ind*m_size 
  
  if str_pos < len(arr) :
    end_pos = min(str_pos + m_size ,len(arr) ) 
    width = 1
    iter_num = 1 

    while width < m_size:
      ##using buffer space in order to avoide data transfer to sub arrays  
      if iter_num % 2 != 0 :
        store_arr = arr
        result_arr = buffer
      else: 
        store_arr = buffer
        result_arr = arr 
    
      
      for i in range(str_pos ,end_pos ,2*width):  
        ##initialising the indexs required to merge two subarrays
        i1 = res_ind = i
        i2 = i + width
        end1 = min( i2 ,end_pos ) 
        end2 = min( i2 + width ,end_pos )
        
        #performing merge operation
        while( i1 < end1 and i2 < end2 ):
          while( i1 < end1 and store_arr[i1] <= store_arr[i2] ):
            result_arr[ res_ind ] = store_arr[i1]
            i1 += 1
            res_ind += 1
          while( i1 < end1 and i2 < end2 and store_arr[i2] < store_arr[i1]  ):
            result_arr[ res_ind ] = store_arr[i2]
            i2 += 1
            res_ind += 1
        while( i1 < end1 ):
          result_arr[res_ind] = store_arr[i1]
          i1 += 1
          res_ind += 1
        while( i2 < end2 ):
          result_arr[res_ind] = store_arr[i2]
          i2 += 1
          res_ind += 1

      #incrementing width and iter  
      width *= 2
      iter_num += 1
    
    
    ##waiting for all the threads to complete sorting its m elements  
    cuda.syncthreads( )
    width = m_size
    
    while( ind%2 == 0 and width < len(arr) ): 
      
      ##using buffer space in order to avoide data transfer to sub arrays  
      if iter_num % 2 != 0 :
        store_arr = arr
        result_arr = buffer                                                                                                   
      else : 
        store_arr = buffer
        result_arr = arr 
      
      ##initialising the indexs required to merge two subarrays 
      i1 = res_ind = str_pos
      i2 = str_pos + width
      end1 = min( i2 ,len(store_arr) ) 
      end2 = min( i2 + width ,len(store_arr) )
      
      ##performing the merge operation
      while( i1 < end1 and i2 < end2 ):
        while( i1 < end1 and store_arr[i1] <= store_arr[i2] ):
          result_arr[ res_ind ] = store_arr[i1]
          i1 += 1
          res_ind += 1
        while( i1 < end1 and i2 < end2 and store_arr[i2] < store_arr[i1]  ):
          result_arr[ res_ind ] = store_arr[i2]
          i2 += 1
          res_ind += 1
      while( i1 < end1 ):
        result_arr[res_ind] = store_arr[i1]
        i1 += 1
        res_ind += 1
      while( i2 < end2 ):
        result_arr[res_ind] = store_arr[i2]
        i2 += 1
        res_ind += 1
      
      #waiting for other threads to complete sorting  
      cuda.syncthreads()
      width *= 2
      ind //= 2
      iter_num += 1
   
    ## transfering the result to main array if needed 
    if iter_num%2 !=0  and ind == 0 : 
      for i in range(0 ,len(arr)):
        arr[i] = buffer[i]