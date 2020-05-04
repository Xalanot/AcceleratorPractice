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
    thrust::device_vector<T> temp(N + 1);

    // compute cumulative sum
    thrust::exclusive_scan(X_d.begin(), X_d.end(), temp.begin());
    temp[N] = X_d.back() + temp[N - 1];

    // compute moving averages from cumulative sum
    thrust::transform(temp.begin() + w, temp.end(), temp.begin(), temp.begin(), minus_and_divide<float>(static_cast<float>(w)));

    checkCudaError(cudaMemcpy(thrust::raw_pointer_cast(result), thrust::raw_pointer_cast(temp.data()), N * float_size, cudaMemcpyDeviceToHost));
}

void simple_moving_average_multi_vs_single(size_t N, int deviceCount)
{
    size_t float_size = sizeof(float);
    float* X_h = static_cast<float*>(malloc(N * float_size));
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 10);
    for (size_t i = 0; i < n; i++)
        X_h[i] = static_cast<float>(dist(rng));

    size_t w = 4;

    float* result_single = static_cast<float*>(malloc( (N - 4) * float_size));
    simple_moving_average_single(X_h, N, w, result_single);
}


int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    //saxpy_multi_vs_single(100000000, deviceCount);
    norm_multi_vs_single(4, deviceCount);

    return 0;
}
