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

    thrust::copy(temp.begin(), temp.begin() + 1, result.begin());
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

        /*if (i == 0)
        {
            for (int j = 0; j < resultSize; ++j)
            {
                if (temp[j] - result_single[j] > 1e-5)
                {
                    std::cout << "WRONG" << std::endl;
                    std::cout << "j: " << j << std::endl;
                    std::cout << "temp: " << temp[j] << std::endl;
                    std::cout << "single: " << result_single[j] << std::endl;
                }
            }
        }*/

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
    simple_moving_average_multi_vs_single(10000, deviceCount);

    return 0;
}
