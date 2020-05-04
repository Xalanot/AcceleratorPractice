#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <chrono>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include "cached_allocator.h"
#include "common_pinned_memory.h"
#include "fallback_allocator.h"

#include "../pml/csvwriter.h"
#include "../pml/measurement.h"

#define DEBUG 0

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
    
    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(hostMemPointer);
    cudaDeviceSynchronize();
    measurementSeries.start();
    thrust::sort(ptr, ptr + numberOfElements);
    cudaDeviceSynchronize();
    measurementSeries.stop();
    assert(thrust::is_sorted(ptr, ptr + numberOfElements));

    cudaFreeHost(hostMemPointer);   
}

template<typename T>
void sort3(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
    fallback_allocator alloc;
    std::cout << "sort3" << std::endl;

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

template <typename T>
void sort4(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
  size_t int_size = sizeof(int);
  if (checkDevice(numberOfElements * int_size))
  {
    return sort3(numberOfElements, measurementSeries);
  }

  thrust::host_vector<int> X_h(numberOfElements);
  thrust::tabulate(X_h.begin(), X_h.end(), get_rand_number(1337, 10 * numberOfElements));

  std::vector<cudaStream_t> streams(3);
  for (auto& stream : streams)
  {
    cudaStreamCreate(&stream);
  }

  std::vector<size_t> sizes(3);
  for (size_t i = 0; i < sizes.size(); ++i)
  {
    if (i != sizes.size() - 1)
    {
      sizes[i] = numberOfElements / sizes.size();
    }
    else
    {
      sizes[i] = (numberOfElements + sizes.size() - 1) / sizes.size();
    }

    std::cout << sizes[i] << std::endl;
  }

  #pragma omp parallel for num_threads(deviceCount) shared(result)
  for (int i = 0; i < 3; ++i)
  {
    thrust::device_vector<int> X_d(sizes[i]);
    checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), thrust::raw_pointer_cast(X_h.data() + i * sizes[0]), sizes[i] * int_size, cudaMemcpyHostToDevice, streams[i]));
    cached_allocator allocator;
    thrust::sort(thrust::cuda::par(allocator).on(streams[i]), X_d.begin(), X_d.end());
    assert(thrust::is_sorted(X_d.begin(), X_d.end()));
  }
  
}

int main(int argc, char *argv[]){
    int iterations = 1;

    std::vector<int> sizes;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort1Times;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort2Times;
    std::vector<MeasurementSeries<std::chrono::milliseconds>> sort3Times;

    size_t N = static_cast<size_t>(1) << 31;
    MeasurementSeries<std::chrono::milliseconds> sort3Series;
    sort4(N, sort3Series);


    /*for (size_t i = 20; i < 25; ++i)
    {
        std::cout << i << std::endl;
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

    csvwriter.write(sizes, sort1Times, sort2Times, sort3Times);*/

    return 0;
}