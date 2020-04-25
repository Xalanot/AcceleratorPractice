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

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y, thrust::device_vector<float>& Z)
{
    thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), saxpy_functor(A));
}

int main(int argc, char** argv)
{
    float a = 2;

    std::vector<int> sizes;
    std::vector<Measurement<std::chrono::seconds>> saxpyFastMeasurements;
    std::vector<Measurement<std::chrono::seconds>> saxpyFast3Measurements;
    std::vector<Measurement<std::chrono::seconds>> saxpyIF_FastMeasurements;

    for (int size = 1; size < 10e9; size *= 10)
    {
        sizes.push_back(size);

        // setup host vector
        thrust::host_vector<float> X_h(size, 1);
        thrust::host_vector<float> Y_h(size);
        thrust::sequence(Y_h.begin(), Y_h.end());

        // copy to device
        thrust::device_vector<float> X_d(X_h);
        thrust::device_vector<float> Y_d(Y_h);

        // saxpy_fast
        Measurement<std::chrono::seconds> saxpyFastMeasurement;
        cudaDeviceSynchronize();
        saxpyFastMeasurement.start();
        saxpy_fast(a, X_d, Y_d);
        cudaDeviceSynchronize();
        saxpyFastMeasurement.stop();
        saxpyFastMeasurements.push_back(saxpyFastMeasurement);

        // saxpy_fast3
        Measurement<std::chrono::seconds> saxpyFast3Measurement;
        thrust::device_vector<float> Z_d(size);
        cudaDeviceSynchronize();
        saxpyFast3Measurement.start();
        saxpy_fast3(a, X_d, Y_d, Z_d);
        cudaDeviceSynchronize();
        saxpyFast3Measurement.stop();
        saxpyFast3Measurements.push_back(saxpyFast3Measurement);


        // copy to host
        X_h = X_d;
        Y_h = Y_d;
    }

    CSVWriter csvwriter("saxpy2.csv");
    std::vector<std::string> headerNames {"size", "hostToDevice", "saxpySlow", "saxpyFast", "deviceToHost"};
    csvwriter.setHeaderNames(std::move(headerNames));
    csvwriter.write(sizes, hostToDeviceMeasurements, saxpySlowMeasurements, saxpyFastMeasurements, deviceToHostMeasurements);

    return 0;
}