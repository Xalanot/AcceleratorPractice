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
    // init index vector
    thrust::device_vector<int> indexVec(10);
    fillRandomIndexVector(indexVec, 20);

    // init value vector
    thrust::device_vector<float> valueVec(10);
    fillRandomValueVector(valueVec, 100);

    print_sparse_vector(indexVec, valueVec);

}

