/*
# Copyright (c) 2011-2012 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.   
*/

#include <iostream>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <thrust/reduce.h>
#include "GpuTimer.h"
#include <cuda_runtime.h>


#define CUDA_SAFE_CALL(call) \
  { \
    cudaError_t err_code = call; \
    if( err_code != cudaSuccess ) { std::cerr << "Error (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(err_code) << std::endl; return 1; } \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// G P U   R E D U C T I O N

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start_memory, start_kernel, start_copyback, end;


__global__ void reduce_kernel( int n, const int *in_buffer, int *out_buffer, const int2 *block_ranges )
{
  // Allocate shared memory inside the block.
  extern __shared__ int s_mem[];

  // The range of data to work with.
  int2 range = block_ranges[blockIdx.x];

  // Compute the sum of my elements.
  
  // TODO: fill-in that section of the code

  // Copy my sum in shared memory.
  //s_mem[threadIdx.x] = my_sum;

  // Make sure all the threads have copied their value in shared memory.
  __syncthreads();

  // Compute the sum inside the block.
  
  // TODO: fill-in that section of the code

  // The first thread of the block stores its result.
  if( threadIdx.x == 0 )
    out_buffer[blockIdx.x] = s_mem[0];
}

int reduce_on_gpu( int n, const int *a_device )
{
  // Compute the size of the grid.
  const int BLOCK_DIM   = 256;
  const int grid_dim    = std::min( BLOCK_DIM, (n + BLOCK_DIM-1) / BLOCK_DIM );
  const int num_threads = BLOCK_DIM * grid_dim;

  // Compute the number of elements per block.
  const int elements_per_block = BLOCK_DIM * ((n + num_threads - 1) / num_threads);

  // Allocate memory for temporary buffers.
  int  *partial_sums = NULL;
  int2 *block_ranges = NULL;

  CUDA_SAFE_CALL( cudaMalloc( (void **) &partial_sums, BLOCK_DIM * sizeof(int ) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **) &block_ranges, grid_dim  * sizeof(int2) ) );

  // Compute the ranges for the blocks.
  int sum = 0;
  int2 *block_ranges_on_host = new int2[grid_dim];
  for( int block_idx = 0 ; block_idx < grid_dim ; ++block_idx )
  {
    block_ranges_on_host[block_idx].x = sum;
    block_ranges_on_host[block_idx].y = std::min( sum += elements_per_block, n );
  }
  CUDA_SAFE_CALL( cudaMemcpy( block_ranges, block_ranges_on_host, grid_dim * sizeof(int2), cudaMemcpyHostToDevice ) );
  delete[] block_ranges_on_host;

  // First round: Compute a partial sum for all blocks.
  reduce_kernel<<<grid_dim, BLOCK_DIM, BLOCK_DIM*sizeof(int)>>>( n, a_device, partial_sums, block_ranges );
  CUDA_SAFE_CALL( cudaGetLastError() );

  // Set the ranges for the second kernel call.
  int2 block_range = make_int2( 0, grid_dim );
  CUDA_SAFE_CALL( cudaMemcpy( block_ranges, &block_range, sizeof(int2), cudaMemcpyHostToDevice ) );

  // Second round: Compute the final sum by summing the partial results of all blocks.
  reduce_kernel<<<1, BLOCK_DIM, BLOCK_DIM*sizeof(int)>>>( grid_dim, partial_sums, partial_sums, block_ranges );
  CUDA_SAFE_CALL( cudaGetLastError() );

  // Read the result from device memory.
  int result;
  CUDA_SAFE_CALL( cudaMemcpy( &result, partial_sums, sizeof(int), cudaMemcpyDeviceToHost ) );

  // Free temporary memory.
  CUDA_SAFE_CALL( cudaFree( block_ranges ) );
  CUDA_SAFE_CALL( cudaFree( partial_sums ) );

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// G P U   R E D U C T I O N :   O P T I M I Z E D   V E R S I O N

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define WARP_SIZE 32

template< int BLOCK_DIM > 
__global__ void reduce_kernel_optimized( int n, const int *in_buffer, int *out_buffer, const int2 *__restrict block_ranges )
{
  // The number of warps in the block.
  const int NUM_WARPS = BLOCK_DIM / WARP_SIZE;

  // Allocate shared memory inside the block.
  __shared__ volatile int s_mem[BLOCK_DIM];

  // The range of data to work with.
  int2 range = block_ranges[blockIdx.x];

  // Warp/lane IDs.
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  // Compute the sum of my elements.
  
  // TODO: fill-in that section of the code

  // Copy my sum in shared memory.
  //s_mem[threadIdx.x] = my_sum;

  // Compute the sum inside each warp.
 
  // TODO: fill-in that section of the code
  
  // Each warp leader stores the result for the warp.
  if( lane_id == 0 )
    // TODO: fill-in that section of the code
  __syncthreads();

  if( warp_id == 0 )
  {
    // Read my value from shared memory and store it in a register.
    //my_sum = s_mem[lane_id];
  
    // Sum the results of the warps.
    
    // TODO: fill-in that section of the code
  }

  // The 1st thread stores the result of the block.
  if( threadIdx.x == 0 )
    return; //To delete and replace with the line below
    //out_buffer[blockIdx.x] = my_sum += s_mem[1];
}

template< int BLOCK_DIM >
int reduce_on_gpu_optimized( int n, const int *a_device )
{
  // Compute the size of the grid.
  const int grid_dim    = std::min( BLOCK_DIM, (n + BLOCK_DIM-1) / BLOCK_DIM );
  const int num_threads = BLOCK_DIM * grid_dim;

  // Compute the number of elements per block.
  const int elements_per_block = BLOCK_DIM * ((n + num_threads - 1) / num_threads);

  // Allocate memory for temporary buffers.
  int  *partial_sums = NULL;
  int2 *block_ranges = NULL;

  CUDA_SAFE_CALL( cudaMalloc( (void **) &partial_sums, BLOCK_DIM * sizeof(int ) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **) &block_ranges, grid_dim  * sizeof(int2) ) );

  // Compute the ranges for the blocks.
  int sum = 0;
  int2 *block_ranges_on_host = new int2[grid_dim];
  for( int block_idx = 0 ; block_idx < grid_dim ; ++block_idx )
  {
    block_ranges_on_host[block_idx].x = sum;
    block_ranges_on_host[block_idx].y = std::min( sum += elements_per_block, n );
  }
  CUDA_SAFE_CALL( cudaMemcpy( block_ranges, block_ranges_on_host, grid_dim * sizeof(int2), cudaMemcpyHostToDevice ) );
  delete[] block_ranges_on_host;

  // First round: Compute a partial sum for all blocks.
  reduce_kernel_optimized<BLOCK_DIM><<<grid_dim, BLOCK_DIM>>>( n, a_device, partial_sums, block_ranges );
  CUDA_SAFE_CALL( cudaGetLastError() );

  // Set the ranges for the second kernel call.
  int2 block_range = make_int2( 0, grid_dim );
  CUDA_SAFE_CALL( cudaMemcpy( block_ranges, &block_range, sizeof(int2), cudaMemcpyHostToDevice ) );

  // Second round: Compute the final sum by summing the partial results of all blocks.
  reduce_kernel_optimized<BLOCK_DIM><<<1, BLOCK_DIM>>>( grid_dim, partial_sums, partial_sums, block_ranges );
  CUDA_SAFE_CALL( cudaGetLastError() );

  // Read the result from device memory.
  int result;
  CUDA_SAFE_CALL( cudaMemcpy( &result, partial_sums, sizeof(int), cudaMemcpyDeviceToHost ) );

  // Free temporary memory.
  CUDA_SAFE_CALL( cudaFree( block_ranges ) );
  CUDA_SAFE_CALL( cudaFree( partial_sums ) );

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// M A I N

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int, char ** )
{

  cudaEventCreate(&start_kernel);
  cudaEventCreate(&end);

  const int NUM_TESTS = 10;

  // The number of elements in the problem.
  const int N = 1024 * 131072;

  std::cout << "Computing a reduction on " << N << " elements" << std::endl;

  // X and Y on the host (CPU).
  int *a_host = new int[N];

  // Make sure the memory got allocated. TODO: free memory.
  if( a_host == NULL )
  {
    std::cerr << "ERROR: Couldn't allocate a_host" << std::endl;
    return 1;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Generate data

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Filling with 1s" << std::endl;

  // Generate pseudo-random data.
  for( int i = 0 ; i < N ; ++i )
    a_host[i] = 1;
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the CPU using 1 thread

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the CPU using 1 CPU thread" << std::endl;
  
  float kernel;
  
  cudaEventRecord(start_kernel);
  // Calculate the reference to compare with the device result.
  int sum = 0;
  for( int i_test = 0 ; i_test < NUM_TESTS ; ++i_test )
  {
    sum = 0;
    for( int i = 0 ; i < N ; ++i )
      sum += a_host[i];
  }

  cudaEventRecord(end);
  cudaEventElapsedTime(&kernel, start_kernel, end);
  
  std::cout << "  Elapsed time: " << kernel / NUM_TESTS << "ms" << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the CPU using several OpenMP threads

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the CPU using " << omp_get_max_threads() << " OpenMP thread(s)" << std::endl;
  
  cudaEventRecord(start_kernel);

  // Calculate the reference to compare with the device result.
  int omp_sum = 0;
  for( int i_test = 0 ; i_test < NUM_TESTS ; ++i_test )
  {
    omp_sum = 0;
#pragma omp parallel shared(omp_sum)
    {
#pragma omp for reduction(+ : omp_sum)
    for( int i = 0 ; i < N ; ++i )
      omp_sum = omp_sum + a_host[i];
    }
  }

  cudaEventRecord(end);
  cudaEventElapsedTime(&kernel, start_kernel, end);
  
  std::cout << "  Elapsed time: " << kernel / NUM_TESTS << "ms" << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the GPU

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // The copy of A on the device (GPU).
  int *a_device = NULL;

  // Allocate A on the device.
  CUDA_SAFE_CALL( cudaMalloc( (void **) &a_device, N*sizeof( int ) ) );

  // Copy A from host (CPU) to device (GPU).
  CUDA_SAFE_CALL( cudaMemcpy( a_device, a_host, N*sizeof( int ), cudaMemcpyHostToDevice ) );



  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the GPU

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the GPU (transfers excluded)" << std::endl;
  
  GpuTimer gpu_timer;
  gpu_timer.Start();

  // Launch the kernel on the GPU.
  int gpu_sum = 0;
  for( int i_test = 0 ; i_test < NUM_TESTS ; ++i_test )
  {
    gpu_sum = reduce_on_gpu( N, a_device );
  }

  gpu_timer.Stop();
  
  std::cout << "  Elapsed time: " << gpu_timer.Elapsed() / NUM_TESTS << "ms" << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the GPU (optimized version)

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the GPU using a tuned version (transfers excluded)" << std::endl;
  
  gpu_timer.Start();

  const int BLOCK_DIM = 256;
  
  // Launch the kernel on the GPU.
  int optim_gpu_sum = 0;
  for( int i_test = 0 ; i_test < NUM_TESTS ; ++i_test )
  {
    optim_gpu_sum = reduce_on_gpu_optimized<BLOCK_DIM>( N, a_device );
  }
  
  gpu_timer.Stop();
  
  std::cout << "  Elapsed time: " << gpu_timer.Elapsed() / NUM_TESTS << "ms" << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Validate results

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Seq    results: red= " << sum << " / sum= " << sum << std::endl;
  std::cout << "OpenMP results: ref= " << sum << " / sum= " << omp_sum << std::endl;
  std::cout << "CUDA   results: ref= " << sum << " / sum= " << gpu_sum << std::endl;
  std::cout << "Optim  results: ref= " << sum << " / sum= " << optim_gpu_sum << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Clean memory

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Free device memory.
  CUDA_SAFE_CALL( cudaFree( a_device ) );
  
  // Free host memory.
  delete[] a_host;

  return 0;
}

