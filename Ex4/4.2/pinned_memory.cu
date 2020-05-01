#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <omp.h>
#include <thrust/sort.h>
#include <chrono>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <cuda_profiler_api.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>


// compile with:  nvcc pinned_memory.cu -O3 -Xcompiler -fopenmp

#define DEBUG 0

// Error handeling of cuda functions
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};


struct not_my_pointer
{
  not_my_pointer(void* p)
    : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() {}

  virtual const char* what() const
  {
    return message.c_str();
  }

private:
  std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator()
  {
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    if(DEBUG)
      std::cout << "cached_allocator::allocate(): num_bytes == " << num_bytes << std::endl;

    char *result = 0;

    // Search the cache for a free block.
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      if(DEBUG)
        std::cout << "cached_allocator::allocate(): found a free block" << std::endl;

      result = free_block->second;

      // Erase from the `free_blocks` map.
      free_blocks.erase(free_block);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with
      // `thrust::cuda::malloc`.
      try
      {
        if(DEBUG)
          std::cout << "cached_allocator::allocate(): allocating new block" << std::endl;

        // Allocate memory and convert the resulting `thrust::cuda::pointer` to
        // a raw pointer.
        result = thrust::cuda::malloc<char>(num_bytes).get();
      }
      catch (std::runtime_error&)
      {
        throw;
      }
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t)
  {
    if(DEBUG)
      std::cout << "cached_allocator::deallocate(): ptr == " << reinterpret_cast<void*>(ptr) << std::endl;

    // Erase the allocated block from the allocated blocks map.
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // Insert the block into the free blocks map.
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {
    if(DEBUG)
      std::cout << "cached_allocator::free_all()" << std::endl;

    // Deallocate all outstanding blocks in both lists.
    for ( free_blocks_type::iterator i = free_blocks.begin()
        ; i != free_blocks.end()
        ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for( allocated_blocks_type::iterator i = allocated_blocks.begin()
       ; i != allocated_blocks.end()
       ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

struct mergePart
{
  size_t size=0;
  unsigned int *ptr;
  bool isSorted=false;
  bool isResult=false;
};

struct payload
{
  int index;
  int numParts;
  struct mergePart *hostParts;
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

void print_device_infos(size_t vecSize, int valueSize, cudaDeviceProp properties, int device){
    std::cout << "Vector size: " << vecSize << std::endl;
    std::cout << "Data size [bytes]: " << (vecSize*valueSize) << std::endl;
  
  if(!properties.unifiedAddressing || !properties.canMapHostMemory)
  {
      std::cout << "Device #" << device 
          << " [" << properties.name << "] does not support memory mapping" << std::endl;
  }
  else
  {
      std::cout << "Device #" << device 
          << " [" << properties.name << "] with " 
          << properties.totalGlobalMem << " bytes of device memory is compatible" << std::endl;
  }
}

void host_merge(int i, int numParts, struct mergePart *hostParts){
  int secMergePartIndex = -1;
  unsigned int *result = NULL;

  if(DEBUG){
    std::cout << "Host parts isSorted: ";
    for (int k = 0; k < numParts; ++k)
      std::cout << "[" << k << "]: " << hostParts[k].isSorted << "   ";
    std::cout << std::endl;
  }

  // on the 2. iteration merge 2 sorted parts
  // on every iteration >2 merge one result and one sorted part
  for(int j = 0; j < numParts; ++j){
    if(j < i && hostParts[j].isResult){
      secMergePartIndex = j;
      break;
    }
  }

  if(secMergePartIndex == -1)
    for(int j = 0; j < numParts; ++j){
      if(j != i && hostParts[j].isSorted){
        secMergePartIndex = j;
        break;
      }
    }
  
  if(secMergePartIndex != -1){
    if(DEBUG)
      std::cout << "start merging part: " << i << " with part: " << secMergePartIndex << std::endl;

    //allocate new space for result array
    checkCudaError(cudaHostAlloc(&result, sizeof(int)*(hostParts[i].size+hostParts[secMergePartIndex].size), 0));

    // merge 2 sorted arrays
    thrust::merge(thrust::omp::par,
      hostParts[i].ptr,
      hostParts[i].ptr+hostParts[i].size,
      hostParts[secMergePartIndex].ptr,
      hostParts[secMergePartIndex].ptr+hostParts[secMergePartIndex].size,
      result);

    // set result to be sorted part, invalidate one of the parts
    checkCudaError(cudaFreeHost(hostParts[i].ptr));
    checkCudaError(cudaFreeHost(hostParts[secMergePartIndex].ptr));
    hostParts[i].ptr = result;
    hostParts[i].isResult = true;
    hostParts[secMergePartIndex].isResult = false;
    if(DEBUG)
      std::cout << "Part " << i << " is result now" << std::endl;
  }
}

int main(int argc, char *argv[]){
    size_t vecSize;
    vecSize = atoll(argv[1]);
    size_t maxMemSize;
    size_t freeMem, totalMem;
    int valueSize = sizeof(int);
    size_t memSize = valueSize*vecSize;
    double timer = 0;
    int device;
    int sufficientMemSize = 1;
    unsigned int maxRandValue = 4294967295;
    cached_allocator alloc;
    //pthread_t mergeThread;

    
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    cudaMemGetInfo(&freeMem, &totalMem);

    // set the max memory size to use by the algorith to
    // the free memory size divided by two because thrust::sort
    // allocates around the same size again. Subtract a small overhead
    // for gpu hardware information pagetable and etc (100Mib).
    maxMemSize = (freeMem/2) - (1<<27);

    if(DEBUG)
      print_device_infos(vecSize, valueSize, properties, device);    
    

    // subtract 100Mib so there is some place for other stuff
    if(memSize > maxMemSize)
        sufficientMemSize = 0;

    if(DEBUG)
        std::cout << "Sufficient memory size: " << sufficientMemSize << std::endl;

   
    // 4.2 sort on gpu with multi-buffering
    cudaStream_t h2dStream, d2hStream, sortStream;
    cudaEvent_t h2dEvent, d2hEvent, sortEvent;
    checkCudaError(cudaStreamCreate(&h2dStream));
    checkCudaError(cudaStreamCreate(&d2hStream));
    checkCudaError(cudaStreamCreate(&sortStream));


    // check if data fits on the gpu in one piece
    int numParts = (memSize/maxMemSize)+1;
    size_t sizeLastPart = memSize - ((numParts-1)*maxMemSize);
    // decrement in case the parts fit perfectly(unlikely)
    if(sizeLastPart == 0)
      --numParts;
    // create array of pointer for each part
    struct mergePart *hostParts = (struct mergePart*)malloc(sizeof(struct mergePart)*numParts);
    struct mergePart *deviceParts = (struct mergePart*)malloc(sizeof(struct mergePart)*numParts);
    
    bool isLastPart = false;

    if(DEBUG){
      std::cout << "numParts: " << numParts << std::endl;
      std::cout << "sizeLastPart: " << sizeLastPart << std::endl;
      std::cout << "size of values [Bytes]: " << valueSize << std::endl;
      std::cout << "max values per part: " << maxMemSize/valueSize << std::endl;
      std::cout << "space left in memory: " << maxMemSize%valueSize << std::endl;
    }


    // fill pointer array with first adress of pinned host memory
    // fill array with random numbers from 0 - vecSize
    #pragma omp parallel for num_threads(numParts)
    for(int i = 0; i < numParts; ++i){
      hostParts[i].ptr = NULL;
      hostParts[i].size = 0;
      hostParts[i].isSorted = false;
      hostParts[i].isResult = false;

      if((sizeLastPart != 0) && (i == numParts-1)){
        hostParts[i].size = (sizeLastPart/valueSize);            
        checkCudaError(cudaHostAlloc(&hostParts[i].ptr, sizeLastPart, 0));
        thrust::tabulate(hostParts[i].ptr, hostParts[i].ptr + hostParts[i].size, get_rand_number(123, maxRandValue));
      }else{
        hostParts[i].size = (maxMemSize/valueSize);            
        checkCudaError(cudaHostAlloc(&hostParts[i].ptr, maxMemSize, 0));
        thrust::tabulate(hostParts[i].ptr, hostParts[i].ptr + hostParts[i].size, get_rand_number(123, maxRandValue)); 
      }
    }


    if(DEBUG){
      cudaMemGetInfo(&freeMem, &totalMem);
      std::cout << "Total Memory | Free Memory "<< std::endl;
      std::cout << totalMem << ", " << freeMem << std::endl;
    }
    
    // start timing after allocation
    auto timer_start = std::chrono::high_resolution_clock::now();

    // start copy, sort, copy-back stream
    for(int i = 0; i < numParts; ++i){
      if(DEBUG)  std::cout << "\niteration: " << i << "\nstart h2d copy" << std::endl;
      
      deviceParts[i].ptr = NULL;
      deviceParts[i].size = 0;
      deviceParts[i].isSorted = false;
      deviceParts[i].isResult = false;

      if((sizeLastPart != 0) && (i == numParts-1)){
        deviceParts[i].size = (sizeLastPart/valueSize);
        checkCudaError(cudaMalloc(&deviceParts[i].ptr, sizeLastPart));
        checkCudaError(cudaMemcpyAsync(deviceParts[i].ptr, hostParts[i].ptr, sizeLastPart, cudaMemcpyHostToDevice, h2dStream));
        isLastPart = true;
      }else{
        deviceParts[i].size = (maxMemSize/valueSize);
        checkCudaError(cudaMalloc(&deviceParts[i].ptr, maxMemSize));
        checkCudaError(cudaMemcpyAsync(deviceParts[i].ptr, hostParts[i].ptr, maxMemSize, cudaMemcpyHostToDevice, h2dStream));
        isLastPart = false;
      }
      
      checkCudaError(cudaEventCreate(&h2dEvent));
      checkCudaError(cudaEventRecord(h2dEvent, h2dStream));          
      
      // wait for copy to complete
      cudaStreamWaitEvent(sortStream, h2dEvent, 0);

      if(DEBUG){
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "Total Memory | Free Memory "<< std::endl;
        std::cout << totalMem << ", " << freeMem << std::endl;
      }
      
      if(DEBUG)  std::cout << "start sorting" << std::endl;         

      if(isLastPart)
        thrust::sort(thrust::cuda::par(alloc).on(sortStream), deviceParts[i].ptr, deviceParts[i].ptr + deviceParts[i].size);
      else
        thrust::sort(thrust::cuda::par(alloc).on(sortStream), deviceParts[i].ptr, deviceParts[i].ptr + deviceParts[i].size);

      checkCudaError(cudaEventCreate(&sortEvent));
      checkCudaError(cudaEventRecord(sortEvent, sortStream));

      // wait for sort to complete
      cudaStreamWaitEvent(d2hStream, sortEvent, 0);          

      if(DEBUG)  std::cout << "start d2h copy" << std::endl;       

      if(isLastPart){
        checkCudaError(cudaMemcpyAsync(hostParts[i].ptr, deviceParts[i].ptr, sizeLastPart, cudaMemcpyDeviceToHost, d2hStream));
      }
      else
        checkCudaError(cudaMemcpyAsync(hostParts[i].ptr, deviceParts[i].ptr, maxMemSize, cudaMemcpyDeviceToHost, d2hStream));

      checkCudaError(cudaEventCreate(&d2hEvent));
      checkCudaError(cudaEventRecord(d2hEvent, d2hStream));
      cudaFree(deviceParts[i].ptr);

      hostParts[i].isSorted = true;

      //host merge
      if(i==0)
        cudaEventSynchronize(d2hEvent);

      // create new thread here and set new payload
      /*
      struct payload p;
      p.numParts = numParts;
      p.hostParts = hostParts;
      p.index = i;
      */

      //status = pthread_create( &mergeThread, NULL, &host_merge, (void*)&p);
      host_merge(i, numParts, hostParts);
    }

    //pthread_join(mergeThread, NULL);

    auto timer_end = std::chrono::high_resolution_clock::now();
    timer = std::chrono::duration<double>(timer_end - timer_start).count();

    std::cout << vecSize << "\t" << timer << "\t" << sufficientMemSize << std::endl;
    
    return 0;
}