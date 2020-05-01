#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <chrono>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#define DEBUG 1

// Error handeling of cuda functions
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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


struct get_rand_number : public thrust::binary_function<void, void, size_t>
{  
  int seed;
  size_t maxRange;
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<size_t> rng_index;

  get_rand_number(int seed, size_t maxRange) {
    seed = seed;
    maxRange = maxRange;
    rng = thrust::default_random_engine(seed);
    rng_index = thrust::uniform_int_distribution<size_t>(0, maxRange);
  }    

  __host__ __device__
  size_t operator()(long x)
  {
    return rng_index(rng);
  }
};

void checkDevice()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // check if the device supports unifiedAdressing and mapHostMemory
    if(!properties.unifiedAddressing || !properties.canMapHostMemory)
    {
        std::cout << "Device #" << device 
            << " [" << properties.name << "] does not support memory mapping" << std::endl;
        exit(1);
    }
    else
    {
        // check if there is enough memory size on the deive, we want to leave 5% left over
        if (properties.totalGlobalMem * 0.95 < memSize)
        {
            std::cout << "Device #" << device <<
                << " [" << properties.name << "] does not have enough memory" << std::endl;
            exit(1);
        }
    }  
}


int main(int argc, char *argv[]){
    size_t vecSize = static_cast<size_t>(1) << 32;
    std::cout << vecSize << std::endl;
    size_t memSize = sizeof(int) * vecSize;
    checkDevice(memSize);
    /*size_t vecSize;
    vecSize = atoll(argv[1]);
    size_t memSize = sizeof(int)*vecSize;
    int *hostMemPointer = NULL;
    double timer0, timer1, timer2;
    int device;
    int sufficientMemSize = 1;
    //int *deviceMemPointer = NULL;

    if(DEBUG){
        std::cout << "Vector size: " << vecSize << std::endl;
        std::cout << "Memory size: " << memSize << std::endl;
    }    

    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // this example requires both unified addressing and memory mapping    
    if(DEBUG)
        if(!properties.unifiedAddressing || !properties.canMapHostMemory)
        {
            std::cout << "Device #" << device 
                << " [" << properties.name << "] does not support memory mapping" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Device #" << device 
                << " [" << properties.name << "] with " 
                << properties.totalGlobalMem << " bytes of device memory is compatible" << std::endl
                << "Datasize is: " << memSize << "\t" << "Max memsize is:" << properties.totalGlobalMem - (100*1024*1024) << std::endl;
        }    
    
    // subtract 100Mib so there is some place for other stuff
    if(memSize > properties.totalGlobalMem - (100*1024*1024))
        sufficientMemSize = 0;

    if(DEBUG)
        std::cout << "Sufficient memory size: " << sufficientMemSize << std::endl;

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