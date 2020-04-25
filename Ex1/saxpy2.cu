#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "pml/csvwriter.h"
#include "pml/measurement.h"


struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(float const& x, float const& y) const
    {
        return a * x + y;
    }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  thrust::device_vector<float> temp(X.size());
   
  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);
    
  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(int argc, char** argv)
{
    float a = 2;

    std::vector<int> sizes;
    std::vector<Measurement<std::chrono::microseconds>> hostToDeviceMeasurements;
    std::vector<Measurement<std::chrono::microseconds>> saxpySlowMeasurements;
    std::vector<Measurement<std::chrono::microseconds>> saxpyFastMeasurements;
    std::vector<Measurement<std::chrono::microseconds>> deviceToHostMeasurements;

    for (int size = 1; size < 10e9; size *= 10)
    {
        sizes.push_back(size);

        // setup host vector
        thrust::host_vector<float> X_h(size, 1);
        thrust::host_vector<float> Y_h(size);
        thrust::sequence(Y_h.begin(), Y_h.end());

        // copy to device
        Measurement<std::chrono::microseconds> hostToDeviceMeasurement;
        cudaDeviceSynchronize();
        hostToDeviceMeasurement.start();
        thrust::device_vector<float> X_d(X_h);
        thrust::device_vector<float> Y_d(Y_h);
        cudaDeviceSynchronize();
        hostToDeviceMeasurement.stop();
        hostToDeviceMeasurements.push_back(hostToDeviceMeasurement);

        // saxpy_slow
        Measurement<std::chrono::microseconds> saxpySlowMeasurement;
        cudaDeviceSynchronize();
        saxpySlowMeasurement.start();
        saxpy_slow(a, X_d, Y_d);
        cudaDeviceSynchronize();
        saxpySlowMeasurement.stop();
        saxpySlowMeasurements.push_back(saxpySlowMeasurement);

        // saxpy_fast
        Measurement<std::chrono::microseconds> saxpyFastMeasurement;
        cudaDeviceSynchronize();
        saxpyFastMeasurement.start();
        saxpy_fast(a, X_d, Y_d);
        cudaDeviceSynchronize();
        saxpyFastMeasurement.stop();
        saxpyFastMeasurements.push_back(saxpyFastMeasurement);

        // copy to host
        Measurement<std::chrono::microseconds> deviceToHostMeasurement;
        cudaDeviceSynchronize();
        deviceToHostMeasurement.start();
        X_h = X_d;
        Y_h = Y_d;
        cudaDeviceSynchronize();
        deviceToHostMeasurement.stop();
        deviceToHostMeasurements.push_back(deviceToHostMeasurement);
    }

    CSVWriter csvwriter("saxpy2.csv");
    std::vector<std::string> headerNames {"size", "hostToDevice", "saxpySlow", "saxpyFast", "deviceToHost"};
    csvwriter.setHeaderNames(std::move(headerNames));
    csvwriter.write(sizes, hostToDeviceMeasurements, saxpySlowMeasurements, saxpyFastMeasurements, deviceToHostMeasurements);

    return 0;
}