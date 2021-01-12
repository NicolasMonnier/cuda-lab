#include <iostream>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <thrust/reduce.h>
#include "GpuTimer.h"
#include <cuda_runtime.h>

cudaEvent_t start_memory, start_kernel, start_copyback, end;

int main( int, char ** )
{
  const int NUM_TESTS = 10;

  // The number of elements in the problem.
  const int N = 512 * 131072;

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

  cudaEventRecord(end)
  cudaEventElapsedTime(&kernel, star_kernel, end);
  
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

  cudaEventRecord(end)
  cudaEventElapsedTime(&kernel, star_kernel, end);
  
  std::cout << "  Elapsed time: " << kernel: / NUM_TESTS << "ms" << std::endl;
  
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

  // Compute on the GPU using Thrust

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the GPU using Thrust (transfers excluded)" << std::endl;
  
  GpuTimer gpu_timer;
  gpu_timer.Start();

  // Launch the kernel on the GPU.
  int thrust_sum = 0;
  for( int i_test = 0 ; i_test < NUM_TESTS ; ++i_test )
  {
    thrust_sum = thrust::reduce( thrust::device_ptr<int>(a_device), thrust::device_ptr<int>(a_device+N) );
  }

  gpu_timer.Stop();
  
  std::cout << "  Elapsed time: " << gpu_timer.Elapsed() / NUM_TESTS << "ms" << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute on the GPU

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl;
  std::cout << "Computing on the GPU (transfers excluded)" << std::endl;
  
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
  std::cout << "OpenMP results: ref= " << sum << " / sum= " << omp_sum << std::endl;
  std::cout << "CUDA   results: ref= " << sum << " / sum= " << gpu_sum << std::endl;
  std::cout << "Thrust results: ref= " << sum << " / sum= " << thrust_sum << std::endl;
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
