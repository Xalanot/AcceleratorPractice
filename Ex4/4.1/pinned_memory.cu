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

#define DEBUG 0
#define ITERATIONS 1

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;


void sort1(size_t N)
{
    size_t memSize = sizeof(int) * N;
    if (!checkDevice(memSize))
    {
      std::cout << "sort1: skipped" << std::endl;
      return;
    }

    int* hostMemPointer = nullptr;
    checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, 0));

    thrust::tabulate(hostMemPointer, hostMemPointer + N, get_rand_number(1337, 10 * N));
    
    // allocate copy vectors
    thrust::device_vector<int> device_vec(N);
    thrust::host_vector<int> host_vec(N);

    cudaDeviceSynchronize();
    auto start = Clock::now();
    for (int i = 0; i < ITERATIONS; ++i)
    {
      // copy to device with hostpointer
      thrust::copy(hostMemPointer, hostMemPointer + N, device_vec.begin());
      // sort on device
      thrust::sort(device_vec.begin(), device_vec.end());
      // transfer back to host
      thrust::host_vector<int> host_vec = device_vec;
    }
    auto end = Clock::now();
    assert(thrust::is_sorted(host_vec.begin(), host_vec.end()));

    auto time = static_cast<Duration>(end - start);

    std::cout << "sort1: " << time.count() / ITERATIONS << std::endl;

    cudaFreeHost(hostMemPointer);
}

void sort2(size_t N)
{
    size_t memSize = sizeof(int) * N;
    if (!checkDevice(memSize))
    {
      std::cout << "sort2: skipped" << std::endl;
      return;
    }

    int* hostMemPointer = nullptr;
    // this time we directly map our data into the adress space of the gpu
    checkCudaError(cudaHostAlloc((void**)&hostMemPointer, memSize, cudaHostAllocMapped));

    thrust::tabulate(hostMemPointer, hostMemPointer + N, get_rand_number(1337, 10 * N));
    
    // if we want to run multiple iterations we need to duplicate the data here
    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(hostMemPointer);
    cudaDeviceSynchronize();
    auto start = Clock::now();
    thrust::sort(ptr, ptr + N);
    cudaDeviceSynchronize();
    auto end = Clock::now();
    assert(thrust::is_sorted(ptr, ptr + N));

    auto time = static_cast<Duration>(end - start);

    std::cout << "sort2: " << time.count() << std::endl;

    cudaFreeHost(hostMemPointer);   
}

void sort3(size_t N)
{
    fallback_allocator alloc;

    // use our special malloc to allocate
    int *raw_ptr = reinterpret_cast<int*>(alloc.allocate(N * sizeof(int)));        

    thrust::cuda::pointer<int> begin = thrust::cuda::pointer<int>(raw_ptr);
    thrust::cuda::pointer<int> end   = begin + N;        

    thrust::tabulate(begin, end, get_rand_number(1337, N));
    cudaDeviceSynchronize();
    auto start = Clock::now();
    try{
        thrust::sort(thrust::cuda::par(alloc), begin, end);
    }
    catch(std::bad_alloc){
        std::cout << "  caught std::bad_alloc from thrust::sort" << std::endl;
    }
    cudaDeviceSynchronize();

    auto time = static_cast<Duration>(Clock::now() - start);

    std::cout << "sort3: " << time.count() << std::endl;

    alloc.deallocate(reinterpret_cast<char*>(raw_ptr), N * sizeof(int));
}

/*template <typename T>
void sort4(size_t numberOfElements, MeasurementSeries<T>& measurementSeries)
{
  size_t int_size = sizeof(int);
  /*if (checkDevice(numberOfElements * int_size))
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

  #pragma omp parallel for num_threads(3)
  for (int i = 0; i < 3; ++i)
  {
    thrust::device_vector<int> X_d(sizes[i]);
    checkCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(X_d.data()), thrust::raw_pointer_cast(X_h.data() + i * sizes[0]), sizes[i] * int_size, cudaMemcpyHostToDevice, streams[i]));
    cached_allocator allocator;
    thrust::sort(thrust::cuda::par(allocator).on(streams[i]), X_d.begin(), X_d.end());
    assert(thrust::is_sorted(X_d.begin(), X_d.end()));
  }
  
}*/

int main(int argc, char ** argv){

    size_t const N = std::stoi(argv[1]);

    sort1(N);
    sort2(N);
    sort3(N);

    return 0;
}