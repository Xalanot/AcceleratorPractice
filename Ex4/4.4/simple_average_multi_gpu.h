#pragma once

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

template <typename T>
struct minus_and_divide : public thrust::binary_function<T,T,T>
{
    T w;

    minus_and_divide(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
        return (a - b) / w;
    }
};

void simple_moving_average_single(thrust::host_vector<float> const& X_h, size_t N, size_t w, thrust::host_vector<float>& result)
{
    thrust::device_vector<float> X_d(X_h);
    
    // allocate storage for cumulative sum
    thrust::device_vector<float> temp(N + 1);

    // compute cumulative sum
    thrust::exclusive_scan(X_d.begin(), X_d.end(), temp.begin());
    temp[N] = X_d.back() + temp[N - 1];

    // compute moving averages from cumulative sum
    thrust::transform(temp.begin() + w, temp.end(), temp.begin(), temp.begin(), minus_and_divide<float>(static_cast<float>(w)));

    thrust::copy(temp.begin(), temp.begin() + result.size(), result.begin());
}

void simple_moving_average_multi(thrust::host_vector<float>& X_h, size_t N, size_t w, thrust::host_vector<float>& result, thrust::host_vector<float> const& result_single, int deviceCount)
{   
    std::vector<DeviceManager> deviceManagers;
    for (int i = 0; i < deviceCount; ++i)
    {
        deviceManagers.emplace_back( DeviceManager{i} );
    }

    #pragma omp parallel for num_threads(deviceCount) shared(result)
    for(int i = 0; i < deviceCount; ++i){

        size_t resultSize = 0;
        size_t deviceSize = 0;

        if ( i == 0)
        {
            resultSize = (N - w + 1 + deviceCount) / deviceCount;
            deviceSize = (N + w) / deviceCount;
            
        }
        else
        {
            resultSize = (N - w + 1) / deviceCount;
            deviceSize = (N + w) / deviceCount - 1;
        }

        size_t ptrOffset = i * ( (N - w) / deviceCount + 1);
        size_t resultOffset = i * (resultSize + 1);

        checkCudaError(cudaSetDevice(i));
        
        thrust::device_vector<float> X_d(X_h.begin() + ptrOffset, X_h.begin() + ptrOffset + deviceSize);
    
        // allocate storage for cumulative sum
        thrust::device_vector<float> temp(deviceSize + 1);

        // compute cumulative sum
        thrust::exclusive_scan(X_d.begin(), X_d.end(), temp.begin());
        temp[deviceSize] = X_d.back() + temp[deviceSize - 1];

        // compute moving averages from cumulative sum
        thrust::transform(temp.begin() + w, temp.end(), temp.begin(), temp.begin(), minus_and_divide<float>(static_cast<float>(w)));

        thrust::copy(temp.begin(), temp.begin() + resultSize, result.begin() + resultOffset);
    }
}

void simple_moving_average_multi_vs_single(size_t N, int deviceCount)
{
    thrust::host_vector<float> X_h(N);
    size_t w = 4;
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 10);
    for (size_t i = 0; i < N; i++)
        X_h[i] = static_cast<float>(dist(rng));

    thrust::host_vector<float> result_single(N - w + 1);
    simple_moving_average_single(X_h, N, w, result_single);

    thrust::host_vector<float> result_multi(N - w + 1);
    simple_moving_average_multi(X_h, N, w, result_multi, result_single, deviceCount);
}