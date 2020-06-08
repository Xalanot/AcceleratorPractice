#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>
#include <chrono>

// This examples compares sorting performance using Array of Structures (AoS)
// and Structure of Arrays (SoA) data layout.  Legacy applications will often
// store data in C/C++ structs, such as MyStruct defined below.  Although 
// Thrust can process array of structs, it is typically less efficient than
// the equivalent structure of arrays layout.  In this particular example,
// the optimized SoA approach is approximately *five times faster* than the
// traditional AoS method.  Therefore, it is almost always worthwhile to
// convert AoS data structures to SoA.

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct MyStruct
{
  int key;
  float value;

  __host__ __device__
    bool operator<(const MyStruct other) const
    {
      return key < other.key;
    }
};

void initialize_keys(thrust::host_vector<int>& keys)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  for(size_t i = 0; i < keys.size(); i++)
    keys[i] = dist(rng);
}


void initialize_keys(thrust::host_vector<MyStruct>& structures)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  for(size_t i = 0; i < structures.size(); i++)
    structures[i].key = dist(rng);
}

void sortAoS(size_t N)
{
    thrust::host_vector<MyStruct> structures_h(N);
    initialize_keys(structures_h);

    cudaDeviceSynchronize();

    auto start = Clock::now();

    thrust::device_vector<MyStruct> structures_d = structures_h;

    thrust::sort(structures_d.begin(), structures_d.end());

    structures_h = structures_d;

    cudaDeviceSynchronize();

    auto end = Clock::now();

    auto duration = static_cast<Duration>(end - start);

    std::cout << "sortAoS: " << duration.count() << std::endl;

    assert(thrust::is_sorted(structures_h.begin(), structures_h.end()));
}

void sortSoA(size_t N)
{
    thrust::host_vector<int>   keys_h(N);
    thrust::host_vector<float> values_h(N);

    initialize_keys(keys_h);

    cudaDeviceSynchronize();

    auto start = Clock::now();

    thrust::device_vector<int> keys_d = keys_h;
    thrust::device_vector<float> values_d = values_h;

    thrust::sort_by_key(keys_d.begin(), keys_d.end(), values_d.begin());

    keys_h = keys_d;
    values_h = values_d;

    cudaDeviceSynchronize();

    auto end = Clock::now();

    auto duration = static_cast<Duration>(end - start);

    std::cout << "sortSoA: " << duration.count() << std::endl;

    assert(thrust::is_sorted(keys_h.begin(), keys_h.end()));
}

struct get_aos
{
  __host__ __device__
  MyStruct operator() (thrust::tuple<int, float> const& tuple)
  {
    MyStruct str;
    str.key = thrust::get<0>(tuple);
    str.value = thrust::get<1>(tuple);
    return str;
  }
};

struct get_soa
{
  __host__ __device__ 
  thrust::tuple<int, float> operator()(MyStruct const& str) 
  {
    return thrust::tuple<int, float> (str.key, str.value);
  }
};

void sort3(size_t N)
{
    thrust::host_vector<MyStruct> structures_h(N);
    thrust::device_vector<MyStruct> structures_d(N);
    thrust::device_vector<int> keys(N);
    thrust::device_vector<float> values(N);

    initialize_keys(structures_h);

    cudaDeviceSynchronize();

    auto start = Clock::now();

    // Copy AoS to SoA on device
    /*auto transform_soa_begin = thrust::make_transform_iterator(structures_h.begin(), get_soa());
    auto transform_soa_end = thrust::make_transform_iterator(structures_h.end(), get_soa());
    thrust::copy(transform_soa_begin, transform_soa_end, thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin())));*/
    structures_d = structures_h;
    thrust::transform(structures_d.begin(), structures_d.end(), keys.begin(), [] __device__ __host__ (MyStruct str) {return str.key;});
    thrust::transform(structures_d.begin(), structures_d.end(), values.begin(), [] __device__ __host__ (MyStruct str) {return str.value;});
    
    // Sort on the device with SoA format
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    auto transform_aos_begin = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin())), get_aos());
    auto transform_aos_end = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(keys.end(), values.end())), get_aos());
    // Transfer data back to host
    thrust::copy(transform_aos_begin, transform_aos_end, structures_h.begin());
    
    cudaDeviceSynchronize();

    auto end = Clock::now();

    auto duration = static_cast<Duration>(end - start);
    std::cout << "sort3: " << duration.count() << std::endl;
    assert(thrust::is_sorted(structures_h.begin(), structures_h.end()));
}

int main(void)
{
  size_t N = 2 * 1024 * 1024;

  // Sort Key-Value pairs using Array of Structures (AoS) storage 
  sortAoS(N);

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  sortSoA(N);

  // Sort 3
  sort3(N);

  return 0;
}

