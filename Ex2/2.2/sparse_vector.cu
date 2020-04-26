#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "advanced_sparse.h"
#include "common_sparse.h"
#include "simple_sparse.h"

#include "../pml/csvwriter.h"
#include "../pml/measurement.h"

int main(void)
{
    int iterations = 100;
    std::vector<int> vectorCount;
    std::vector<MeasurementSeries<std::chrono::microseconds>> oldTimes;
    std::vector<MeasurementSeries<std::chrono::microseconds>> newTimes;


    for (int number_vectors = 2; number_vectors <= 10; ++number_vectors)
    {
        vectorCount.push_back(number_vectors);
        // generate index vectors
        std::vector<thrust::device_vector<int>> indexVectors(number_vectors);
        for (size_t i = 0; i < indexVectors.size(); ++i)
        {
            thrust::device_vector<int> tmpIndex(10);
            fillRandomIndexVector(tmpIndex, 20);
            indexVectors[i] = tmpIndex;
        }

        // generate value vectors
        std::vector<thrust::device_vector<float>> valueVectors(number_vectors);
        for (size_t i = 0; i < valueVectors.size(); ++i)
        {
            thrust::device_vector<float> tmpValue(10);
            fillRandomValueVector(tmpValue, 10);
            valueVectors[i] = tmpValue;
        }

        // use old method
        MeasurementSeries<std::chrono::microseconds> measurementOld;
        thrust::device_vector<int> result_index_old;
        thrust::device_vector<float> result_value_old;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            measurementOld.start();
            sum_sparse_vectors(indexVectors, valueVectors, result_index_old, result_value_old);
            cudaDeviceSynchronize();
            measurementOld.stop();
        }
        oldTimes.push_back(measurementOld);

        // use new method
        MeasurementSeries<std::chrono::microseconds> measurementNew;
        thrust::device_vector<int> result_index_new;
        thrust::device_vector<float> result_value_new;
        for (int i = 0; i < iterations; ++i)
        {
            cudaDeviceSynchronize();
            measurementNew.start();
            sum_multiple_sparse_vectors(indexVectors, valueVectors, result_index_new, result_value_new);
            cudaDeviceSynchronize();
            measurementNew.stop();
        }
        newTimes.push_back(measurementNew);
        
        if (checkResults(result_index_old, result_value_old, result_index_new, result_value_new))
        {
            std::cout << "result is wrong" << std::endl;
        }
    }

    CSVWriter csvwriter("sparse.csv");
    std::vector<std::string> headerNames {"vector count", "old method", "new method"};
    csvwriter.setHeaderNames(std::move(headerNames));

    csvwriter.write(vectorCount, oldTimes, newTimes);

}

