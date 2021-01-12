#pragma once
#include <cuda_runtime_api.h>

class GpuTimer
{
  cudaEvent_t start, stop;

public:
  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
  }

  void Start()
  {
    cudaEventRecord(start);
  }

  void Stop()
  {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

