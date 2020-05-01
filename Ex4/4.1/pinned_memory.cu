#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <chrono>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include "common_pinned_memory.h"

#include "../pml/csvwriter.h"
#include "../pml/measurement.h"

#define DEBUG 0

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

template<typename T>
void sort1(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
    size_t memSize = sizeof(int) * numberOfElements;
    checkDevice(memSize);

    int* hostMemPointer = nullptr;
    checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, 0));

    thrust::tabulate(hostMemPointer, hostMemPointer + numberOfElements, get_rand_number(1337, 10 * numberOfElements));
    
    cudaDeviceSynchronize();
    measurementSeries.start();
    // copy to device with hostpointer
    thrust::device_vector<int> device_vec(hostMemPointer, hostMemPointer + numberOfElements);
    // sort on device
    thrust::sort(device_vec.begin(), device_vec.end());
    // transfer back to host
    thrust::host_vector<int> host_vec = device_vec;
    cudaDeviceSynchronize();
    measurementSeries.stop();

    cudaFreeHost(hostMemPointer);
}

template<typename T>
void sort2(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
    size_t memSize = sizeof(int) * numberOfElements;
    checkDevice(memSize);

    int* hostMemPointer = nullptr;
    checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, cudaHostAllocPortable));

    thrust::tabulate(hostMemPointer, hostMemPointer + numberOfElements, get_rand_number(1337, 10 * numberOfElements));
    
    cudaDeviceSynchronize();
    measurementSeries.start();
    // copy to device with hostpointer
    thrust::device_vector<int> device_vec(hostMemPointer, hostMemPointer + numberOfElements);
    // sort on device
    thrust::sort(device_vec.begin(), device_vec.end());
    // transfer back to host
    thrust::host_vector<int> host_vec = device_vec;
    cudaDeviceSynchronize();
    measurementSeries.stop();

    cudaFreeHost(hostMemPointer);   
}

template<typename T>
void sort3(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
    fallback_allocator alloc;

    // use our special malloc to allocate
    int *raw_ptr = reinterpret_cast<int*>(alloc.allocate(numberOfElements * sizeof(int)));        

    thrust::cuda::pointer<int> begin = thrust::cuda::pointer<int>(raw_ptr);
    thrust::cuda::pointer<int> end   = begin + numberOfElements;        

    thrust::tabulate(begin, end, get_rand_number(1337, numberOfElements));
    cudaDeviceSynchronize();
    measurementSeries.start();
    try{
        thrust::sort(thrust::cuda::par(alloc), begin, end);
    }
    catch(std::bad_alloc){
        std::cout << "  caught std::bad_alloc from thrust::sort" << std::endl;
    }

    cudaDeviceSynchronize();
    measurementSeries.stop();

    alloc.deallocate(reinterpret_cast<char*>(raw_ptr), numberOfElements * sizeof(int));
}

int main(int argc, char *argv[]){
    int iterations = 10;

    std::vector<int> sizes;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort1Times;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort2Times;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort3Times;

    for (size_t i = 20; i < 33; ++i)
    {
        size_t numberOfElements = static_cast<size_t>(1) << i;
        if (!checkDevice(sizeof(int) * i))
        {
            break;
        }

        sizes.push_back(i);

        MeasurementSeries<std::chrono::milliseconds> sort1Series;
        for (int j = 0; j < iterations; ++j)
        {
            sort1(numberOfElements, sort1Series);
        }
        sort1Times.push_back(sort1Series);

        MeasurementSeries<std::chrono::milliseconds> sort2Series;
        for (int j = 0; j < iterations; ++j)
        {
            sort2(numberOfElements, sort2Series);
        }
        sort2Times.push_back(sort2Series);

        MeasurementSeries<std::chrono::milliseconds> sort3Series;
        for (int j = 0; j < iterations; ++j)
        {
            sort3(numberOfElements, sort3Series);
        }
        sort3Times.push_back(sort3Series);
    }

    CSVWriter csvwriter("pinned_memory.csv");
    csvwriter.setHeaderNames( {"size", "sort1", "sort2", "sort3"});

    csvwriter.write(sizes, sort1Times, sort2Times, sort3Times);

    return 0;
}