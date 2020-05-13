#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <vector>

template<typename T>
thrust::device_vector<T> concatInSingleVector(std::vector<thrust::device_vector<T>> const& vectors)
{
    // calculate final size
    size_t size = 0;
    for (auto const& vec : vectors)
    {
        size += vec.size();
    }

    thrust::device_vector<T> returnVec(size);

    size_t offset = 0;
    for (auto const& vec: vectors)
    {
        thrust::copy(vec.begin(), vec.end(), returnVec.begin() + offset);
        offset += vec.size();
    }

    return returnVec;
}

template<typename IndexVector>
size_t countUniqueElements(IndexVector const& vector)
{
    int currentIndex = -1;
    size_t count = 0;
    for (int i = 0; i < vector.size(); ++i)
    {
        int index = vector[i];
        if (index > currentIndex)
        {
            count++;
            currentIndex = index;
        }
    }

    return count;
}


template<typename IndexVectors,
         typename ValueVectors,
         typename IndexVector,
         typename ValueVector>
void sum_multiple_sparse_vectors(IndexVectors const& indexVectors,
                                 ValueVectors const& valueVectors,
                                 IndexVector& C_index,
                                 ValueVector& C_value)
{
    typedef typename IndexVector::value_type  IndexType;
    typedef typename ValueVector::value_type  ValueType;

    // first we want add all index and value vectors in a single vector
    IndexVector tmp_index = concatInSingleVector(indexVectors);
    ValueVector tmp_value = concatInSingleVector(valueVectors);

    // sort by keys
    thrust::sort_by_key(thrust::device, tmp_index.begin(), tmp_index.end(), tmp_value.begin());

    // get unique index size
    size_t unique_index_size = size_t C_size = thrust::inner_product(tmp_index.begin(), tmp_index.end() - 1,
                                          tmp_index.begin() + 1,
                                          size_t(0),
                                          thrust::plus<size_t>(),
                                          thrust::not_equal_to<IndexType>()) + 1;

    // allocate space for output
    C_index.resize(unique_index_size);
    C_value.resize(unique_index_size);

    // sum values with the same index
    thrust::reduce_by_key(tmp_index.begin(), tmp_index.end(),
                          tmp_value.begin(),
                          C_index.begin(),
                          C_value.begin(),
                          thrust::equal_to<IndexType>(),
                          thrust::plus<ValueType>());
}
