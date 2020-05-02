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
#include <stdlib.h>
#include <cassert>
#include <thrust/system/cuda/memory.h>

#include "common_multi_gpu.h"

#include "../pml/measurement.h"

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_single(float a, float* X_h, float* Y_h, float* Z_h, size_t N)
{
    size_t float_size = sizeof(float);
    checkCudaError(cudaSetDevice(0));

    thrust::device_vector<float> X_d(N);
    thrust::device_vector<float> Y_d(N);
    thrust::device_vector<float> Z_d(N);

    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(X_d.data()), thrust::raw_pointer_cast(X_h), N * float_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(Y_d.data()), thrust::raw_pointer_cast(Y_h), N * float_size, cudaMemcpyHostToDevice));

    Measurement<std::chrono::microseconds> measurement;
    cudaDeviceSynchronize();
    measurement.start();
    thrust::transform(X_d.begin(), X_d.end(), Y_d.begin(), Z_d.begin(), saxpy_functor(a));
    cudaDeviceSynchronize();
    measurement.stop();
    std::cout << "total time single: " << measurement << " milliseconds" << std::endl;

    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(Z_h), thrust::raw_pointer_cast(Z_d.data()), N * float_size, cudaMemcpyDeviceToHost));
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

        checkCudaError(cudaEventRecord(deviceManagers[i].start, deviceManagers[i].transformStream));
        thrust::transform(thrust::cuda::par.on(deviceManagers[i].transformStream), X_d.begin(), X_d.end(), Y_d.begin(), Z_d.begin(), saxpy_functor(a));
        checkCudaError(cudaEventRecord(deviceManagers[i].stop, deviceManagers[i].transformStream));
        checkCudaError(cudaEventSynchronize(deviceManagers[i].stop));
        checkCudaError(cudaEventElapsedTime(&deviceManagers[i].myTime, deviceManagers[i].start, deviceManagers[i].stop));

        checkCudaError(cudaEventRecord(deviceManagers[i].transformEvent, deviceManagers[i].transformStream));
        cudaStreamWaitEvent(deviceManagers[i].transformStream, deviceManagers[i].transformEvent, 0);

        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(Z_h + i * deviceSize), thrust::raw_pointer_cast(Z_d.data()), deviceSize * float_size, cudaMemcpyDeviceToHost, deviceManagers[i].d2hStream));

        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].d2hStream));
        cudaStreamWaitEvent(deviceManagers[i].d2hStream, deviceManagers[i].copyEvent, 0); 
    }

    float totalTime = 0;
    for (auto const& deviceManager : deviceManagers)
    {
        totalTime += deviceManager.myTime;
    }
    std::cout << "total time multi: " << totalTime << " milliseconds" << std::endl;
}

void saxpy_multi_vs_single(size_t N, int deviceCount)
{
    size_t float_size = sizeof(float);

    float* X_h = nullptr;
    float* Y_h = nullptr;

    // allocate memory on host device
    checkCudaError(cudaHostAlloc(&X_h, float_size * N, 0));
    checkCudaError(cudaHostAlloc(&Y_h, float_size * N, 0));

    // fill data with random values
    thrust::tabulate(X_h, X_h + N, get_rand_number(42, 10));
    thrust::tabulate(Y_h, Y_h + N, get_rand_number(1337, 10));

    // saxpy_single
    float a = 2.f;
    float* Z_h_single = static_cast<float*>(malloc(N * float_size));
    saxpy_single(a, X_h, Y_h, Z_h_single, N);
    std::cout << "after single" << std::endl;

    // saxpy_multi
    float* Z_h_multi = static_cast<float*>(malloc(N * float_size));
    saxpy_multi(2.f, X_h, Y_h, Z_h_multi, N, deviceCount);

    for (int i = 0; i < N; ++i)
    {
        if (abs(Z_h_single[i] - Z_h_multi[i]) > 1e-5)
        {
            std::cout << "wrong result" << std::endl;
            std::cout << "single: " << Z_h_single[i] << std::endl;
            std::cout << "multiple: " << Z_h_multi[i] << std::endl;
        }
    }

    cudaFreeHost(X_h);
    cudaFreeHost(Y_h);
    free(Z_h_single);
    free(Z_h_multi);
}



int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    saxpy_multi_vs_single(4, deviceCount);

    return 0;
}
