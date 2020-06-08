#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>

// This examples compares sorting performance using Array of Structures (AoS)
// and Structure of Arrays (SoA) data layout.  Legacy applications will often
// store data in C/C++ structs, such as MyStruct defined below.  Although 
// Thrust can process array of structs, it is typically less efficient than
// the equivalent structure of arrays layout.  In this particular example,
// the optimized SoA approach is approximately *five times faster* than the
// traditional AoS method.  Therefore, it is almost always worthwhile to
// convert AoS data structures to SoA.

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
    measurementSeries.start();

    thrust::device_vector<MyStruct> structures_d = structures_h;

    thrust::sort(structures_d.begin(), structures_d.end());

    structures_h = structures_d;

    cudaDeviceSynchronize();
    measurementSeries.stop();

    assert(thrust::is_sorted(structures_h.begin(), structures_h.end()));
}

void sortSoA(size_t N)
{
    thrust::host_vector<int>   keys_h(N);
    thrust::host_vector<float> values_h(N);

    initialize_keys(keys_h);

    cudaDeviceSynchronize();
    measurementSeries.start();

    thrust::device_vector<int> keys_d = keys_h;
    thrust::device_vector<float> values_d = values_h;

    thrust::sort_by_key(keys_d.begin(), keys_d.end(), values_d.begin());

    keys_h = keys_d;
    values_h = values_d;

    cudaDeviceSynchronize();
    measurementSeries.stop();

    assert(thrust::is_sorted(keys_h.begin(), keys_h.end()));
}

void sort3(size_t N)
{
    thrust::host_vector<MyStruct> structures_h(N);
    thrust::device_vector<MyStruct> structures_d(N);
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);

    initialize_keys(structures_h);

    cudaDeviceSynchronize();
    measurementSeries.start();

    // Copy SoA to device
    //structures_d = structures_h;
    // Transfer data to AoS on device
    thrust::transform(structures_h.begin(), structures_h.end(), keys.begin(), [] __device__ __host__ (MyStruct str) {return str.key;});
    thrust::transform(structures_h.begin(), structures_h.end(), values.begin(), [] __device__ __host__ (MyStruct str) {return str.value;});
    
    // Sort on the device with SoA format
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    //assert(thrust::is_sorted(keys.begin(), keys.end()));
    
    // Transfer data back to host
    /*thrust::transform(keys.begin(), keys.end(), structures_d.begin(), [] __device__ __host__ (int key) 
                      {MyStruct str;
                       str.key = key;
                       return str;});*/
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(keys.end(), values.end())),
                      structures_h.begin(),
                      [] __device__ __host__ (thrust::tuple<int, float> t)
                      {MyStruct str;
                      str.key = thrust::get<0>(t);
                      str.value = thrust::get<1>(t);
                      return str;});
    
    cudaDeviceSynchronize();
    measurementSeries.stop();
    assert(thrust::is_sorted(structures_h.begin(), structures_h.end()));
}

int main(void)
{
  size_t N = 2 * 1024 * 1024;

  // Sort Key-Value pairs using Array of Structures (AoS) storage 
  //sortAoS(N, AoSSeries);

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  //sortSoA(N, SoASeries);

  // Sort 3
  sort3(N);

  return 0;
}

