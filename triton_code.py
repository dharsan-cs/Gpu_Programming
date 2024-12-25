import torch 
import triton 
import triton.language as tl

@triton.jit
def triton_softmax(a_ptr ,n_col :tl.constexpr ):
  #starting index of row corresponding to the block  
  str_pos = tl.program_id(0) * n_col
  ## getting ptrs corresponding to remaining elements in the row  
  offsets = str_pos + tl.arange(0 ,n_col )
  ##loading the data  
  row = tl.load( pointer = a_ptr + offsets )
  ##finding max of all elements  
  max_val = tl.max( row ,axis = 0 )
  ## row[i] = e**(max_val - row[i])
  numerator = tl.exp(max_val - row )
  ##denominator
  denominator = tl.sum(numerator)
  ##storing the results in a
  tl.store( pointer = a_ptr + offsets ,value = numerator/denominator )

@triton.jit
def triton_vec_addition( x_vec ,y_vec ,output_vec ,block_size:tl.constexpr ,n_element:tl.constexpr ): 
  ## getting the block index along the x axis  
  str_index = block_size * tl.program_id( 0 )
  
  ## getting index of remaining elements in the block
  offset = str_index + tl.arange(0 ,block_size)
  
  ## getting mask to prevent unwanted data acces  
  mask = offset < n_element
  
  ## loading the required data into memory 
  x = tl.load( pointer = x_vec + offset ,mask = mask ,other = 0.0 ) 
  y = tl.load( pointer = y_vec + offset ,mask = mask ,other = 0.0 ) 
  
  ## storing the output in the output vector 
  tl.store( pointer = output_vec + offset ,value = x + y ,mask = mask)