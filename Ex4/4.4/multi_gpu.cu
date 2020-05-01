#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <thrust/random.h>
#include <cuda.h>
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

int main(int argc, char **argv)
{
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));
    std::cout << "device count: " << deviceCount << std::endl;
}
