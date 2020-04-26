#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

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

void fillRandomIndexVector(thrust::device_vector<int>& indexVec, int maxIndex)
{
    thrust::minstd_rand rng_i;
    thrust::uniform_int_distribution<int> rng_index(0, maxIndex);

    std::set<int> indexSet;
    while (indexSet.size() != indexVec.size())
    {
        int tmpIndex = rng_index(rng_i);
        indexSet.insert(tmpIndex);
    }

    thrust::copy(indexSet.begin(), indexSet.end(), indexVec.begin());
}

template<typename T>
void fillRandomValueVector(thrust::device_vector<T>& valueVec, int maxValue)
{
    thrust::minstd_rand rng_i;
    thrust::uniform_real_distribution<float> rng_value(0.0, 1000.0);
    for (size_t i = 0; i < valueVec.size(); ++i)
    {
        valueVec[i] = rng_value(rng_i);
    }
}

bool checkResults(thrust::device_vector<int> const& indexVector1, 
                  thrust::device_vector<float> const& valueVector1,
                  thrust::device_vector<int> const& indexVector2,
                  thrust::device_vector<float> const& valueVector2)
{
    if (indexVector1 == indexVector2 && valueVector1 == valueVector2)
    {
        return true;
    }

    return false;
}