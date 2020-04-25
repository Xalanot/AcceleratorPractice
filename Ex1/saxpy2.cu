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
    int iterations = 100;

    std::vector<int> sizes;
    std::vector<MeasurementSeries<std::chrono::microseconds>> hostToDevice;
    std::vector<MeasurementSeries<std::chrono::microseconds>> saxpySlow;
    std::vector<MeasurementSeries<std::chrono::microseconds>> saxpyFast;
    std::vector<MeasurementSeries<std::chrono::microseconds>> deviceToHost;

    for (int size = 1; size < 10e6; size *= 10)
    {
        sizes.push_back(size);

        // setup host vector
        thrust::host_vector<float> X_h(size, 1);
        thrust::host_vector<float> Y_h(size);
        thrust::sequence(Y_h.begin(), Y_h.end());

        // copy to device
        MeasurementSeries<std::chrono::microseconds> hostToDeviceMeasurementSeries;
        thrust::device_vector<float> X_d;
        thrust::device_vector<float> Y_d;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            hostToDeviceMeasurementSeries.start();
            X_d = X_h;
            Y_d = Y_h;
            cudaDeviceSynchronize();
            hostToDeviceMeasurementSeries.stop();
        }
        hostToDevice.push_back(hostToDeviceMeasurementSeries);

        // saxpy_slow
        MeasurementSeries<std::chrono::microseconds> saxpySlowMeasurementSeries;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            saxpySlowMeasurementSeries.start();
            saxpy_slow(a, X_d, Y_d);
            cudaDeviceSynchronize();
            saxpySlowMeasurementSeries.stop();
        }
        saxpySlow.push_back(saxpySlowMeasurementSeries);

        // saxpy_fast
        MeasurementSeries<std::chrono::microseconds> saxpyFastMeasurementSeries;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            saxpyFastMeasurementSeries.start();
            saxpy_fast(a, X_d, Y_d);
            cudaDeviceSynchronize();
            saxpyFastMeasurementSeries.stop();
        }
        saxpyFast.push_back(saxpyFastMeasurementSeries);

        // copy to host
        MeasurementSeries<std::chrono::microseconds> deviceToHostMeasurementSeries;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            deviceToHostMeasurementSeries.start();
            X_h = X_d;
            Y_h = Y_d;
            cudaDeviceSynchronize();
            deviceToHostMeasurementSeries.stop();
        }
        deviceToHost.push_back(deviceToHostMeasurementSeries);
    }

    CSVWriter csvwriter("saxpy2.csv");
    std::vector<std::string> headerNames {"size", "hostToDevice", "saxpySlow", "saxpyFast", "deviceToHost"};
    csvwriter.setHeaderNames(std::move(headerNames));
    csvwriter.write(sizes, hostToDevice, saxpySlow, saxpyFast, deviceToHost);

    return 0;
}