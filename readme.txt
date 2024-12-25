
# GPU Programming Repository

This repository contains a collection of GPU programming scripts and examples. These scripts demonstrate various GPU-based and CPU-based programming concepts, optimizations, and algorithms using Python. Below is an overview of the files included in this repository.

## File Descriptions

### Matrix Multiplication
1. **`normal_matmul.py`**
   Implements a basic matrix multiplication algorithm.

2. **`coalesced_matmul.py`**
   Shows matrix multiplication with coalesced memory access for better GPU efficiency.

3. **`tile_matmul.py`**
   Demonstrates tiled matrix multiplication to optimize memory access using l1 cache in streaming multiprocessor.

4. **`one_dim_tiling.py`**
   Uses the same memory optimization used in tile_matmul with reduced number of threads.

5. **`two_dim_tiling.py`**
   Uses the same memory optimization used in tile_matmul with very smaller number of threads compared one_dim_tiling.

### Strided Memory Access
6. **`coalesced_strided_access.py`**
   to demonstrate the time difference between strided memory access and coalesced memory access

### Sorting Algorithms
7. **`merge_sort_cpu.py`**
   CPU implementation of the Merge Sort algorithm for comparison with the GPU version.

8. **`merge_sort_gpu.py`**
   GPU implementation of the Merge Sort algorithm, showcasing parallelism and efficiency.

### Graph Algorithms
9. **`dijkstra_cpu_code.py`**
   CPU implementation of Dijkstra's shortest path algorithm.

10. **`dijkstra_gpu_code.py`**
    GPU implementation of Dijkstra's algorithm, showcasing parallelism and efficiency.

### CUDA Streams
11. **`cuda_streams.py`**
    Example showcasing the use of CUDA streams for overlapping computation and data transfer.

### Triton Programming
12. **`triton_code.py`**
    A script written using Triton, a language designed for efficient GPU programming.

