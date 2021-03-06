#pragma once

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#define DEBUG 1

// Error handeling of cuda functions
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct get_rand_number : public thrust::binary_function<void, void, size_t>
{  
  int seed;
  size_t maxRange;
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<size_t> rng_index;

  get_rand_number(int seed, size_t maxRange) {
    seed = seed;
    maxRange = maxRange;
    rng = thrust::default_random_engine(seed);
    rng_index = thrust::uniform_int_distribution<size_t>(0, maxRange);
  }    

  __host__ __device__
  size_t operator()(long x)
  {
    return rng_index(rng);
  }
};

size_t bytesToMBytes(size_t bytes)
{
    return bytes >> 20;
}

size_t bytesToGBytes(size_t bytes)
{
    return bytes >> 30;
}

bool checkDevice(size_t memSize)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // check if the device supports unifiedAdressing and mapHostMemory
    if(!properties.unifiedAddressing || !properties.canMapHostMemory)
    {
        if (DEBUG)
        {
            std::cout << "Device #" << device 
            << " [" << properties.name << "] does not support memory mapping" << std::endl;
        }
        return false;
    }
    else
    {
        if (DEBUG)
        {
            std::cout << "Checking for " << bytesToGBytes(memSize) << " GB" << std::endl;
        }
        // check if there is enough memory size on the deive, we want to leave 5% left over
        if (properties.totalGlobalMem * 0.95 < 2 * memSize)
        {
            if (DEBUG)
            {
                std::cout << "Device #" << device
                    << " [" << properties.name << "] does not have enough memory" << std::endl;
                std::cout << "There is " << bytesToMBytes(2 * memSize - properties.totalGlobalMem * 0.95) << "MB too few bytes of memory" << std::endl;
            }
            return false;
        }
    }

    return true;  
}