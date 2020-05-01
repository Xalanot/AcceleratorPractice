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

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

struct DeviceManager{
    DeviceManager(int i)
    {
        cudaSetDevice(i);

        checkCudaError(cudaStreamCreateWithFlags(&h2dStream, cudaStreamNonBlocking));
        checkCudaError(cudaStreamCreateWithFlags(&d2hStream, cudaStreamNonBlocking));  
        checkCudaError(cudaStreamCreateWithFlags(&transformStream, cudaStreamNonBlocking));

        checkCudaError(cudaEventCreate(&transformEvent));
        checkCudaError(cudaEventCreate(&copyEvent));
    }

    cudaStream_t h2dStream, d2hStream, transformStream;
    cudaEvent_t transformEvent, copyEvent;
};

int main(int argc, char **argv)
{
    size_t float_size = sizeof(float);
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));
    
    float* X_h = nullptr;
    float* Y_h = nullptr;
    size_t N = 4;

    // allocate memory on host device
    checkCudaError(cudaHostAlloc(&X_h, float_size * N, 0));
    checkCudaError(cudaHostAlloc(&Y_h, float_size * N, 0));

    // fill data with random values
    thrust::tabulate(X_h, X_h + N, get_rand_number(42, 10));
    thrust::tabulate(Y_h, Y_h + N, get_rand_number(1337, 10));

    std::cout << "X0: " << X_h[0] << std::endl;
    std::cout << "Y0: " << Y_h[0] << std::endl;
    std::cout << "X3: " << X_h[3] << std::endl;
    std::cout << "Y3: " << Y_h[3] << std::endl;

    std::vector<DeviceManager> deviceManagers;
    for (int i = 0; i < deviceCount; ++i)
    {
        deviceManagers.emplace_back( DeviceManager{i} );
    }

    #pragma omp parallel num_threads(deviceCount)
    for (int i = 0; i < deviceCount; ++i)
    {
        std::cout << "i: " << i << std::endl;
        // set the device
        checkCudaError(cudaSetDevice(i));

        // copy data on the device
        thrust::device_vector<float> X_d(2);
        thrust::device_vector<float> Y_d(2);

        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), X_h + i * 2, 2, cudaMemcpyHostToDevice, deviceManagers[i].h2dStream));
        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(Y_d.data()), Y_h + i * 2, 2, cudaMemcpyHostToDevice, deviceManagers[i].h2dStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].h2dStream));
        cudaStreamWaitEvent(deviceManagers[i].h2dStream, deviceManagers[i].copyEvent, 0);

        //checkCudaError(cudaEventRecord(deviceManagers[i].start, deviceManagers[i].transformStream));
        thrust::transform(thrust::cuda::par.on(deviceManagers[i].transformStream), X_d.begin(), X_d.end(), Y_d.begin(), Y_d.begin(), saxpy_functor(2));
        //checkCudaError(cudaEventRecord(deviceManagers[i].stop, deviceManagers[i].transformStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].transformEvent, deviceManagers[i].transformStream));
        cudaStreamWaitEvent(deviceManagers[i].transformStream, deviceManagers[i].transformEvent, 0);        

        checkCudaError(cudaMemcpyAsync(Y_h + i * 2, thrust::raw_pointer_cast(Y_d.data()), 2, cudaMemcpyDeviceToHost, deviceManagers[i].d2hStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].d2hStream));
        cudaStreamWaitEvent(deviceManagers[i].d2hStream, deviceManagers[i].copyEvent, 0); 
    }

    std::cout << "Z0: " << Y_h[0] << std::endl;
    std::cout << "Z3: " << Y_h[3] << std::endl;

    return 0;
}
