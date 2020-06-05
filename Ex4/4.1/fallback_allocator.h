#pragma once

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