#include <thrust/random.h>
#include <cuda.h>
#include <thread>
#include <thrust/execution_policy.h>
#include <chrono>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>
#include <thrust/system/cuda/memory.h>

// Error handeling of cuda functions
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};

struct get_rand_number : public thrust::binary_function<void, void, float>
{  
  int seed;
  size_t maxRange;
  thrust::default_random_engine rng;
  thrust::random::uniform_real_distribution<float> rng_index;

  get_rand_number(int seed, size_t maxRange) {
    seed = seed;
    maxRange = maxRange;
    rng = thrust::default_random_engine(seed);
    rng_index = thrust::uniform_real_distribution<float>(0, maxRange);
  }    

  __host__ __device__
  float operator()(long x)
  {
    return rng_index(rng);
  }
};

struct DeviceManager{
    DeviceManager(int i)
    {
        checkCudaError(cudaSetDevice(i));

        checkCudaError(cudaStreamCreateWithFlags(&h2dStream, cudaStreamNonBlocking));
        checkCudaError(cudaStreamCreateWithFlags(&d2hStream, cudaStreamNonBlocking));  
        checkCudaError(cudaStreamCreateWithFlags(&transformStream, cudaStreamNonBlocking));

        checkCudaError(cudaEventCreate(&transformEvent));
        checkCudaError(cudaEventCreate(&copyEvent));
    }

    cudaStream_t h2dStream, d2hStream, transformStream;
    cudaEvent_t transformEvent, copyEvent;
};