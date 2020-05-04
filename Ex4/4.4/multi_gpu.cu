#include "norm_multi_gpu.h"
#include "saxpy_multi_gpu.h"

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

void simple_moving_average_single(float* X_h, size_t N, size_t w, float* result)
{
    size_t float_size = sizeof(float);
    thrust::device_vector<float> X_d(N);
    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(X_d.data()), thrust::raw_pointer_cast(X_h), N * float_size, cudaMemcpyHostToDevice));
    
    // allocate storage for cumulative sum
    thrust::device_vector<float> temp(N + 1);

    // compute cumulative sum
    thrust::exclusive_scan(X_d.begin(), X_d.end(), temp.begin());
    temp[N] = X_d.back() + temp[N - 1];

    // compute moving averages from cumulative sum
    thrust::transform(temp.begin() + w, temp.end(), temp.begin(), temp.begin(), minus_and_divide<float>(static_cast<float>(w)));

    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(result), thrust::raw_pointer_cast(temp.data()), N * float_size, cudaMemcpyDeviceToHost));
}

void simple_moving_average_multi(float *X_h, size_t N, size_t w, float* result, int deviceCount)
{   
    deviceCount = 1; 
    std::vector<DeviceManager> deviceManagers;
    for (int i = 0; i < deviceCount; ++i)
    {
        deviceManagers.emplace_back( DeviceManager{i} );
    }

    size_t float_size = sizeof(float);

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
        
        thrust::device_vector<float>X_d(deviceSize);
        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), X_h, 1 * float_size, cudaMemcpyDefault, deviceManagers[i].h2dStream));

        // wait for copy to complete
        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].h2dStream));
        cudaStreamWaitEvent(deviceManagers[i].h2dStream, deviceManagers[i].copyEvent, 0);
    
        // allocate storage for cumulative sum
        thrust::device_vector<float> temp(deviceSize);

        // compute cumulative sum
        thrust::exclusive_scan(thrust::cuda::par.on(deviceManagers[i].transformStream), X_d.begin(), X_d.end(), temp.begin());
        temp[temp.size()] = X_d.back() + temp[temp.size() - 1];

        // compute moving averages from cumulative sum
        thrust::transform(thrust::cuda::par.on(deviceManagers[i].transformStream), temp.begin() + w, temp.end(), temp.begin(), temp.begin(), minus_and_divide<float>(static_cast<float>(w)));

        //checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(result), thrust::raw_pointer_cast(temp.data()), 1 * float_size, cudaMemcpyDeviceToHost));
        /*
        checkCudaError(cudaEventSynchronize(myDevices[i].stop));
        checkCudaError(cudaEventElapsedTime(&myDevices[i].myTime, myDevices[i].start, myDevices[i].stop));
        */
    }
}

void simple_moving_average_multi_vs_single(size_t N, int deviceCount)
{
    size_t float_size = sizeof(float);
    float* X_h = static_cast<float*>(malloc(N * float_size));
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 10);
    for (size_t i = 0; i < N; i++)
        X_h[i] = static_cast<float>(dist(rng));

    size_t w = 4;

    float* result_single = static_cast<float*>(malloc( (N - w + 1) * float_size));
    simple_moving_average_single(X_h, N, w, result_single);

    float* result_multi = static_cast<float*>(malloc( (N - w + 1) * float_size));
    simple_moving_average_multi(X_h, N, w, result_single, deviceCount);

    for (int i = 0; i < N - w + 1; ++i)
    {
        if (result_single[i] - result_multi[i] > 1e-5)
        {
            std::cout << "wrong result at: " << i << std::endl;
            std::cout << "single: " << result_single[i] << std::endl;
            std::cout << "multi: " << result_multi[i] << std::endl;
        }
    }
}


int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    //saxpy_multi_vs_single(100000000, deviceCount);
    //norm_multi_vs_single(4, deviceCount);
    simple_moving_average_multi_vs_single(10, deviceCount);

    return 0;
}
