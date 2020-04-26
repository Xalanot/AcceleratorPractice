#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "advanced_sparse.h"
#include "simple_sparse.h"

template <typename IndexVector,
          typename ValueVector>
void print_sparse_vector(const IndexVector& A_index,
                         const ValueVector& A_value)
{
    // sanity test
    assert(A_index.size() == A_value.size());

    for(size_t i = 0; i < A_index.size(); i++)
        std::cout << "(" << A_index[i] << "," << A_value[i] << ") ";
    std::cout << std::endl;
}

int main(void)
{
    // initialize sparse vector A with 4 elements
    thrust::device_vector<int>   A_index(4);
    thrust::device_vector<float> A_value(4);
    A_index[0] = 2;  A_value[0] = 10;
    A_index[1] = 3;  A_value[1] = 60;
    A_index[2] = 5;  A_value[2] = 20;
    A_index[3] = 8;  A_value[3] = 40;
    
    // initialize sparse vector B with 6 elements
    thrust::device_vector<int>   B_index(6);
    thrust::device_vector<float> B_value(6);
    B_index[0] = 1;  B_value[0] = 50;
    B_index[1] = 2;  B_value[1] = 30;
    B_index[2] = 4;  B_value[2] = 80;
    B_index[3] = 5;  B_value[3] = 30;
    B_index[4] = 7;  B_value[4] = 90;
    B_index[5] = 8;  B_value[5] = 10;

    // initalize sparse vector D with 5 elements
    thrust::device_vector<int>   D_index(5);
    thrust::device_vector<float> D_value(5);
    D_index[0] = 1;  D_value[0] = 50;
    D_index[1] = 2;  D_value[1] = 30;
    D_index[2] = 5;  D_value[2] = 80;
    D_index[3] = 6;  D_value[3] = 30;
    D_index[4] = 7;  D_value[4] = 90;

    // compute sparse vector C = A + B
    thrust::device_vector<int>   C_index;
    thrust::device_vector<float> C_value;

    std::vector<thrust::device_vector<int>> vectors_index {A_index, B_index, D_index};
    std::vector<thrust::device_vector<float>> vectors_value {A_value, B_value, D_value};
    thrust::device_vector<int>   C_index2;
    thrust::device_vector<float> C_value2;
    
    sum_sparse_vectors(vectors_index, vectors_value, C_index, C_value);
    sum_multiple_sparse_vectors(vectors_index, vectors_value, C_index2, C_value2);

    std::cout << "Computing C = A + B for sparse vectors A and B" << std::endl;
    std::cout << "A "; print_sparse_vector(A_index, A_value);
    std::cout << "B "; print_sparse_vector(B_index, B_value);
    std::cout << "C "; print_sparse_vector(C_index, C_value);
    std::cout << "C2 "; print_sparse_vector(C_index2, C_value2);
}

