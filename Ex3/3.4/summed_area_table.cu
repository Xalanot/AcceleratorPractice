#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "../pml/measurement.h"
#include "../pml/csvwriter.h"

// This example computes a summed area table using segmented scan
// http://en.wikipedia.org/wiki/Summed_area_table



// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
  size_t n;
  
  __host__ __device__
  row_index(size_t _n) : n(_n) {}

  __host__ __device__
  size_t operator()(size_t i)
  {
      return i / n;
  }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
  thrust::counting_iterator<size_t> indices(0);
  
  thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
     thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
     src.begin(),
     dst.begin());
}


// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);

  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(n)),
     thrust::make_transform_iterator(indices, row_index(n)) + d_data.size(),
     d_data.begin(),
     d_data.begin());
}

void scan_old(size_t m, size_t n, thrust::device_vector<int>& data)
{
  // [step 1] scan horizontally
  scan_horizontally(n, data);

  // [step 2] transpose array
  thrust::device_vector<int> temp(m * n);
  transpose(m, n, data, temp);

  // [step 3] scan transpose horizontally
  scan_horizontally(m, temp);

  // [step 4] transpose the transpose
  transpose(n, m, temp, data);
}

thrust::device_vector<int> generateTransposeMap(size_t m, size_t n)
{
    thrust::device_vector<int> transposeMap(m * n);
    for (size_t i = 0; i < m * n; ++i)
    {
        size_t j = i / m;
        size_t k = i % m;

        transposeMap[i] = n * k + j;
    }

    return transposeMap;
}

// scan the rows of an M-by-N array
template <typename T>
void scan_vertically(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);
  //auto transposeMap = generateTransposeMap(m, n);
  auto mapBegin = thrust::make_transform_iterator(indices, transpose_index(m,n));

  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(m)),
     thrust::make_transform_iterator(indices, row_index(m)) + d_data.size(),
     thrust::make_permutation_iterator(d_data.begin(), mapBegin),
     thrust::make_permutation_iterator(d_data.begin(), mapBegin));
}

void scan_new(size_t m, size_t n, thrust::device_vector<int>& data)
{
    // [step 1] scan horizontally
    scan_horizontally(n, data);

    // [step 2] scan vertically
    scan_vertically(m, n, data);
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::host_vector<T> h_data = d_data;

  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    std::cout << "\n";
  }
}

int main(void)
{
  int iterations = 10;

  std::vector<size_t> mVec;
  std::vector<size_t> nVec;
  std::vector<MeasurementSeries<std::chrono::microseconds>> oldTimes;
  std::vector<MeasurementSeries<std::chrono::microseconds>> newTimes;
  // first run
  size_t m = 1000; // number of rows
  mVec.push_back(m);
  size_t n = 1000; // number of columns
  nVec.push_back(n);

  MeasurementSeries<std::chrono::microseconds> oldMeasurement;
  for (int i = 0; i < iterations; ++i)
  {
      thrust::device_vector<int> data(m * n, 1);
      cudaDeviceSynchronize();
      oldMeasurement.start();
      scan_old(m, n, data);
      cudaDeviceSynchronize();
      oldMeasurement.stop();
  }
  oldTimes.push_back(oldMeasurement);


  MeasurementSeries<std::chrono::microseconds> newMeasurement;
  for (int i = 0; i < iterations; ++i)
  {
      thrust::device_vector<int> data(m * n, 1);
      cudaDeviceSynchronize();
      newMeasurement.start();
      scan_new(m, n, data);
      cudaDeviceSynchronize();
      newMeasurement.stop();
  }
  newTimes.push_back(newMeasurement);

  // sanity check
  thrust::device_vector<int> data_old(m * n, 1);
  thrust::device_vector<int> data_new(m * n, 1);
  scan_old(m, n, data_old);
  scan_new(m, n, data_new);
  if (data_old != data_new)
  {
      std::cout << "wrong result" << std::endl;
  }
  
  // second run
  m = 100000; // number of rows
  mVec.push_back(m);
  n = 10; // number of columns
  nVec.push_back(n);

  MeasurementSeries<std::chrono::microseconds> oldMeasurement2;
  for (int i = 0; i < iterations; ++i)
  {
      thrust::device_vector<int> data(m * n, 1);
      cudaDeviceSynchronize();
      oldMeasurement2.start();
      scan_old(m, n, data);
      cudaDeviceSynchronize();
      oldMeasurement2.stop();
  }
  oldTimes.push_back(oldMeasurement2);


  MeasurementSeries<std::chrono::microseconds> newMeasurement2;
  for (int i = 0; i < iterations; ++i)
  {
      thrust::device_vector<int> data(m * n, 1);
      cudaDeviceSynchronize();
      newMeasurement2.start();
      scan_new(m, n, data);
      cudaDeviceSynchronize();
      newMeasurement2.stop();
  }
  newTimes.push_back(newMeasurement2);

  // sanity check
  thrust::device_vector<int> data_old2(m * n, 1);
  thrust::device_vector<int> data_new2(m * n, 1);
  scan_old(m, n, data_old2);
  scan_new(m, n, data_new2);
  if (data_old2 != data_new2)
  {
      std::cout << "wrong result" << std::endl;
  }

  CSVWriter csvwriter("summed_area_table.csv");
  csvwriter.setHeaderNames( {"m", "n", "old times", "new times"} );

  csvwriter.write(mVec, nVec, oldTimes, newTimes);


  return 0;
}
