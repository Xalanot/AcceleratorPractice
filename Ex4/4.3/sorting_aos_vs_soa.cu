#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>

#include "../pml/csvwriter.h"
#include "../pml/measurement.h"

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


void initialize_keys(thrust::device_vector<MyStruct>& structures)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<MyStruct> h_structures(structures.size());

  for(size_t i = 0; i < h_structures.size(); i++)
    h_structures[i].key = dist(rng);

  structures = h_structures;
}

template<typename T>
void sortAoS(size_t N, MeasurementSeries<T>& measurementSeries)
{
    thrust::host_vector<MyStruct> structures_h(N);
    initialize_keys(structures_h);

    cudaDeviceSynchronize();
    measurementSeries.start();

    thrust::device_vector<MyStruct> structures_d = structures_h;

    thrust::sort(structures_d.begin(), structures_d.end());

    structures_h = Structures_d;

    cudaDeviceSynchronize();
    measurementSeries.stop();

    assert(thrust::is_sorted(structures_h.begin(), structures_h.end()));
}

template<typename T>
void sortSoA(size_t N, MeasurementSeries<T>& measurementSeries)
{
    thrust::host_vector<int>   keys_h(N);
    thrust::host_vector<float> values_h(N);

    initialize_keys(keys_h);

    cudaDeviceSynchronize();
    measurementSeries.start();

    thrust::device_vector<int> keys_d = keys_h;
    thrust::device_vector<float> values_d = values_h;

    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());

    keys_d = keys_h;
    values_d = values_h;

    cudaDeviceSynchronize();
    measurementSeries.stop();

    assert(thrust::is_sorted(keys_h.begin(), keys_h.end()));
}

int main(void)
{
  typedef std::chrono::microseconds time;
  size_t N = 2 * 1024 * 1024;
  int iterations = 10;

  // Sort Key-Value pairs using Array of Structures (AoS) storage 
  MeasurementSeries<time> AoSSeries;
  for (int i = 0; i < iterations; ++i)
  {
      sortAoS(N, AoSSeries);
  }

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  MeasurementSeries<time> SoASeries;
  for (int i = 0; i < iterations; ++i)
  {
      sortSoA(N, SoASeries);
  }

  std::cout << "aos: " << AoSSeries << std::endl;
  std::cout << "soa: " << SoASeries << std::endl;

  return 0;
}

