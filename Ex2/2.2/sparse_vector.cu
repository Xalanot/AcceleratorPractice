#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <iostream>
#include <vector>

#include "advanced_sparse.h"
#include "common_sparse.h"
#include "simple_sparse.h"

int main(void)
{
    for (int number_vectors = 2; number_vectors <= 10; ++number_vectors)
    {
        // generate index vectors
        std::vector<thrust::device_vector<int>> indexVectors(number_vectors);
        for (size_t i = 0; i < indexVectors.size(); ++i)
        {
            thrust::device_vector<int> tmpIndex(10000);
            fillRandomIndexVector(tmpIndex, 1000000);
            indexVectors[i] = tmpIndex;
        }

        // generate value vectors
        std::vector<thrust::device_vector<float>> valueVectors(number_vectors);
        for (size_t i = 0; i < valueVectors.size(); ++i)
        {
            thrust::device_vector<float> tmpValue(10000);
            fillRandomValueVector(tmpValue, 100);
            valueVectors[i] = tmpValue;
        }

        // use old method
        thrust::device_vector<int> result_index_old;
        thrust::device_vector<float> result_value_old;
        sum_sparse_vectors(indexVectors, valueVectors, result_index_old, result_value_old);

        // use new method
        // use old method
        thrust::device_vector<int> result_index_new;
        thrust::device_vector<float> result_value_new;
        sum_multiple_sparse_vectors(indexVectors, valueVectors, result_index_new, result_value_new);
        
        if (checkResults(result_index_old, result_value_old, result_index_new, result_value_new))
        {
            std::cout << "result is right" << std::endl;
        }
        else 
        {
            std:: cout << "wrong" << std::endl;
        }
    }

}

