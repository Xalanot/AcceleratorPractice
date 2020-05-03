#include "norm_multi_gpu.h"
#include "saxpy_multi_gpu.h"



int main(int argc, char **argv)
{ 
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount));

    //saxpy_multi_vs_single(100000000, deviceCount);
    norm_multi_vs_single(4, deviceCount);

    return 0;
}
