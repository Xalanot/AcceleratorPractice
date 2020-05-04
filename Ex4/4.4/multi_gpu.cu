#include "norm_multi_gpu.h"
#include "saxpy_multi_gpu.h"
#include "simple_average_multi.gpu."

int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    //saxpy_multi_vs_single(100000000, deviceCount);
    //norm_multi_vs_single(4, deviceCount);
    simple_moving_average_multi_vs_single(10000, deviceCount);

    return 0;
}
