#include "saxpy_multi_gpu.h"

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

float norm_multi_gpu(float *X_h, size_t N, int deviceCount)
{    
    std::vector<DeviceManager> deviceManagers;
    for (int i = 0; i < deviceCount; ++i)
    {
        deviceManagers.emplace_back( DeviceManager{i} );
    }

    size_t deviceSize = N / deviceCount;
    size_t float_size = sizeof(float);
   
    float result = 0;
    square<float> unary_op;
    thrust::plus<float> binary_op;

    #pragma omp parallel for num_threads(deviceCount) shared(result)
    for(int i = 0; i < deviceCount; ++i){

        checkCudaError(cudaSetDevice(i));
        
        thrust::device_vector<float>X_d(deviceSize);
        checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), X_h + i * deviceCount, deviceCount * float_size, cudaMemcpyDefault, deviceManagers[i].h2dStream));

        // wait for copy to complete
        checkCudaError(cudaEventRecord(deviceManagers[i].copyEvent, deviceManagers[i].h2dStream));
        cudaStreamWaitEvent(deviceManagers[i].h2dStream, deviceManagers[i].copyEvent, 0);

        checkCudaError(cudaEventRecord(deviceManagers[i].start, deviceManagers[i].transformStream));
        result += thrust::transform_reduce(thrust::cuda::par.on(deviceManagers[i].transformStream), X_d.begin(), X_d.end(), unary_op, 0, binary_op);
        checkCudaError(cudaEventRecord(deviceManagers[i].stop, deviceManagers[i].transformStream));

        /*
        checkCudaError(cudaEventSynchronize(myDevices[i].stop));
        checkCudaError(cudaEventElapsedTime(&myDevices[i].myTime, myDevices[i].start, myDevices[i].stop));
        */
    }
    cudaDeviceSynchronize();

    return result;
}

void norm_multi_vs_single(size_t N, int deviceCount)
{
    size_t float_size = sizeof(float);

    float* X_h = nullptr;
    checkCudaError(cudaHostAlloc(&X_h, float_size * N, 0));
    thrust::tabulate(X_h, X_h + N, get_rand_number(43, 10));

    for (int i = 0; i < N; ++i)
    {
        std::cout << X_h[i] << std::endl;
    }

    float result = norm_multi_gpu(X_h, N, deviceCount);

    std::cout << "result: " << result << std::endl;
}

int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    //saxpy_multi_vs_single(100000000, deviceCount);
    norm_multi_vs_single(4, deviceCount);

    return 0;
}
