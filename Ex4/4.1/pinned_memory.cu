#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <chrono>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include "common_pinned_memory.h"


class fallback_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;

    // allocate's job to is allocate host memory as a functional fallback when cudaMalloc fails
    char *allocate(std::ptrdiff_t n)
    {
      char *result = 0;

      // attempt to allocate device memory
      if(cudaMalloc(&result, n) == cudaSuccess)
      {
        if(DEBUG)
            std::cout << "  allocated " << n << " bytes of device memory" << std::endl;
      }
      else
      {
        // reset the last CUDA error
        cudaGetLastError();

        // attempt to allocate pinned host memory
        void *h_ptr = 0;
        if(cudaMallocHost(&h_ptr, n) == cudaSuccess)
        {
          // attempt to map host pointer into device memory space
          if(cudaHostGetDevicePointer(&result, h_ptr, 0) == cudaSuccess)
          {
            if(DEBUG)
                std::cout << "  allocated " << n << " bytes of pinned host memory (fallback successful)" << std::endl;
          }
          else
          {
            // reset the last CUDA error
            cudaGetLastError();

            // attempt to deallocate buffer
            if(DEBUG)
                std::cout << "  failed to map host memory into device address space (fallback failed)" << std::endl;
            cudaFreeHost(h_ptr);

            throw std::bad_alloc();
          }
        }
        else
        {
          // reset the last CUDA error
          cudaGetLastError();
          if(DEBUG)
            std::cout << "  failed to allocate " << n << " bytes of memory (fallback failed)" << std::endl;

          throw std::bad_alloc();
        }
      }

      return result;
    }

    // deallocate's job to is inspect where the pointer lives and free it appropriately
    void deallocate(char *ptr, size_t n)
    {
      void *raw_ptr = thrust::raw_pointer_cast(ptr);

      // determine where memory resides
      cudaPointerAttributes	attributes;

      if(cudaPointerGetAttributes(&attributes, raw_ptr) == cudaSuccess)
      {
        // free the memory in the appropriate way
        if(attributes.memoryType == cudaMemoryTypeHost)
        {
          cudaFreeHost(raw_ptr);
        }
        else
        {
          cudaFree(raw_ptr);
        }
      }
    }
};

void sort1(size_t numberOfElements)
{
    size_t memSize = sizeof(int) * numberOfElements;
    checkDevice(memSize);

    int* hostMemPointer = nullptr;
    checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, 0));

    thrust::tabulate(hostMemPointer, hostMemPointer + numberOfElements, get_rand_number(1337, vecSize));
    
    // copy to device with hostpointer
    thrust::device_vector<int> device_vec(hostMemPointer, hostMemPointer + numberOfElements);
    for (int i = 0; i < device_vec.size(); ++i)
    {
        std::cout << device_vec[i] << std::endl;
    }
    std::cout << "sort" << std::endl
    // sort on device
    thrust::sort(device_vec.begin(), device_vec.end());
    // transfer back to host
    for (int i = 0; i < device_vec.size(); ++i)
    {
        std::cout << device_vec[i] << std::endl;
    }
    thrust::host_vector<int> host_vec = device_vec;


    cudaFreeHost(hostMemPointer);
}

int main(int argc, char *argv[]){
    size_t numberOfElements= static_cast<size_t>(1) << 2;
    sort1(numberOfElements);
    /*size_t vecSize;
    vecSize = atoll(argv[1]);
    size_t memSize = sizeof(int)*vecSize;
    int *hostMemPointer = NULL;
    double timer0, timer1, timer2;
    int device;
    int sufficientMemSize = 1;
    //int *deviceMemPointer = NULL;

    // 4.1.1 sort on gpu with copy from host and transfer back
    if(sufficientMemSize)
    {
        checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, 0));

        thrust::tabulate(hostMemPointer, hostMemPointer + vecSize, get_rand_number(123, vecSize));

        cudaDeviceSynchronize();
        auto timer_start = std::chrono::high_resolution_clock::now();
        
        // copy to device with hostpointer
        thrust::device_vector<int>device_vec(hostMemPointer, hostMemPointer + vecSize);
        // sort on device
        thrust::sort(device_vec.begin(), device_vec.end());
        // transfer back to host
        thrust::host_vector<int>host_vec = device_vec;

        auto timer_end = std::chrono::high_resolution_clock::now();
        timer0 = std::chrono::duration<double>(timer_end - timer_start).count();

        cudaFreeHost(hostMemPointer);
    }

    // 4.1.2 sort on gpu with no copy
    if(sufficientMemSize)
    {
        hostMemPointer = NULL;
        // allocate space on host in cuda adress space
        checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, cudaHostAllocPortable));

        thrust::tabulate(hostMemPointer, hostMemPointer + vecSize, get_rand_number(123, vecSize));
        
        // set device vector to pointer of host memory in cuda adress space
        thrust::device_vector<int>device_vec_unf(hostMemPointer, hostMemPointer + vecSize);
        
        cudaDeviceSynchronize();
        auto timer_start = std::chrono::high_resolution_clock::now();
        // sort on device
        thrust::sort(device_vec_unf.begin(), device_vec_unf.end());
        auto timer_end = std::chrono::high_resolution_clock::now();

        timer1 = std::chrono::duration<double>(timer_end - timer_start).count();

        cudaFreeHost(hostMemPointer);
    }

    // 4.1.3 sort on host and device memory with fallback allocator
    {
        fallback_allocator alloc;

        // use our special malloc to allocate
        int *raw_ptr = reinterpret_cast<int*>(alloc.allocate(vecSize * sizeof(int)));        

        thrust::cuda::pointer<int> begin = thrust::cuda::pointer<int>(raw_ptr);
        thrust::cuda::pointer<int> end   = begin + vecSize;        

        thrust::tabulate(begin, end, get_rand_number(123, vecSize));

        cudaDeviceSynchronize();
        auto timer_start = std::chrono::high_resolution_clock::now();
        try{
            thrust::sort(thrust::cuda::par(alloc), begin, end);
        }
        catch(std::bad_alloc){
            std::cout << "  caught std::bad_alloc from thrust::sort" << std::endl;
        }
        auto timer_end = std::chrono::high_resolution_clock::now();

        timer2 = std::chrono::duration<double>(timer_end - timer_start).count();

        alloc.deallocate(reinterpret_cast<char*>(raw_ptr), vecSize * sizeof(int));
    }

    std::cout << vecSize << "\t" << timer0 << "\t" << timer1 << "\t" << timer2 << "\t" << std::endl; */

    return 0;
}