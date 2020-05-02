#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>
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

#include "common_multi_gpu.h"

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_multi_vs_single(size_t N, int deviceCount)
{
    size_t float_size = sizeof(float);

    float* X_h = nullptr;
    float* Y_h = nullptr;
    float* Z_h_multi = nullptr;

    // allocate memory on host device
    checkCudaError(cudaHostAlloc(&X_h, float_size * N, 0));
    checkCudaError(cudaHostAlloc(&Y_h, float_size * N, 0));
    Z_h_multi = static_cast<float*>(malloc(N * float_size))

    // fill data with random values
    thrust::tabulate(X_h, X_h + N, get_rand_number(42, 10));
    thrust::tabulate(Y_h, Y_h + N, get_rand_number(1337, 10));

    std::cout << "X0: " << X_h[0] << std::endl;
    std::cout << "Y0: " << Y_h[0] << std::endl;
    std::cout << "X3: " << X_h[3] << std::endl;
    std::cout << "Y3: " << Y_h[3] << std::endl;

    // call saxpy_multi
    saxpy_multi(2.f, X_h, Y_h, Z_h, N, deviceCount);

    std::cout << "Z0: " << Z_h[0] << std::endl;
    std::cout << "Z3: " << Z_h[3] << std::endl;
}

void saxpy_multi(float a, float* X_h, float* Y_h, float* Z_h, size_t N, int deviceCount)
{
    std::vector<DeviceManager> deviceManagers;
    for (int i = 0; i < deviceCount; ++i)
    {
        deviceManagers.emplace_back( DeviceManager{i} );
    }

    size_t deviceSize = N / deviceCount;
    size_t float_size = sizeof(float);

    #pragma omp parallel for num_threads(deviceCount)
    for (int i = 0; i < deviceCount; ++i)
    {
        // set the device
        checkCudaError(cudaSetDevice(i));

        // copy data on the device
        thrust::device_vector<float> X_d(deviceSize);
        thrust::device_vector<float> Y_d(deviceSize);
        thrust::device_vector<float> Z_d(deviceSize);

        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), thrust::raw_pointer_cast(X_h + i * deviceSize), deviceSize * float_size, cudaMemcpyHostToDevice, deviceManagers[i].h2dStream));
        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(Y_d.data()), thrust::raw_pointer_cast(Y_h + i * deviceSize), deviceSize * float_size, cudaMemcpyHostToDevice, deviceManagers[i].h2dStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].h2dStream));
        cudaStreamWaitEvent(deviceManagers[i].h2dStream, deviceManagers[i].copyEvent, 0);

        //checkCudaError(cudaEventRecord(deviceManagers[i].start, deviceManagers[i].transformStream));
        thrust::transform(thrust::cuda::par.on(deviceManagers[i].transformStream), X_d.begin(), X_d.end(), Y_d.begin(), Z_d.begin(), saxpy_functor(a));
        //checkCudaError(cudaEventRecord(deviceManagers[i].stop, deviceManagers[i].transformStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].transformEvent, deviceManagers[i].transformStream));
        cudaStreamWaitEvent(deviceManagers[i].transformStream, deviceManagers[i].transformEvent, 0);        
        //std::cout << "copy back" << std::endl;
        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(Z_h + i * deviceSize), thrust::raw_pointer_cast(Z_d.data()), deviceSize * float_size, cudaMemcpyDeviceToHost, deviceManagers[i].d2hStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].d2hStream));
        cudaStreamWaitEvent(deviceManagers[i].d2hStream, deviceManagers[i].copyEvent, 0); 
    }
}

int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    saxpy_multi_vs_single(2, deviceCount);

    return 0;
}
