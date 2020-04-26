#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

template <typename IndexVector1,
          typename ValueVector1,
          typename IndexVector2,
          typename ValueVector2,
          typename IndexVector3,
          typename ValueVector3>
void sum_sparse_vectors(const IndexVector1& A_index,
                        const ValueVector1& A_value,
                        const IndexVector2& B_index,
                        const ValueVector2& B_value,
                              IndexVector3& C_index,
                              ValueVector3& C_value)
{
    typedef typename IndexVector3::value_type  IndexType;
    typedef typename ValueVector3::value_type  ValueType;

    // sanity test
    assert(A_index.size() == A_value.size());
    assert(B_index.size() == B_value.size());

    size_t A_size = A_index.size();
    size_t B_size = B_index.size();

    // allocate storage for the combined contents of sparse vectors A and B
    IndexVector3 temp_index(A_size + B_size);
    ValueVector3 temp_value(A_size + B_size);

    // merge A and B by index
    thrust::merge_by_key(A_index.begin(), A_index.end(),
                         B_index.begin(), B_index.end(),
                         A_value.begin(),
                         B_value.begin(),
                         temp_index.begin(),
                         temp_value.begin());
    
    // compute number of unique indices
    size_t C_size = thrust::inner_product(temp_index.begin(), temp_index.end() - 1,
                                          temp_index.begin() + 1,
                                          size_t(0),
                                          thrust::plus<size_t>(),
                                          thrust::not_equal_to<IndexType>()) + 1;

    // allocate space for output
    C_index.resize(C_size);
    C_value.resize(C_size);

    // sum values with the same index
    thrust::reduce_by_key(temp_index.begin(), temp_index.end(),
                          temp_value.begin(),
                          C_index.begin(),
                          C_value.begin(),
                          thrust::equal_to<IndexType>(),
                          thrust::plus<ValueType>());
}

template <typename IndexVectors,
          typename ValueVectors,
          typename IndexVector,
          typename ValueVector>
void sum_sparse_vectors(IndexVectors const& indexVectors,
                        ValueVectors const& valueVectors,
                        IndexVector& C_index,
                        ValueVector& C_value)
{
    sum_sparse_vectors(indexVectors[0], valueVectors[0], indexVectors[1], valueVectors[1], C_index, C_value);

    for (size_t i = 2; i < indexVectors.size(); ++i)
    {
        sum_sparse_vectors(indexVectors[i], valueVectors[i], C_index, C_value, C_index, C_value);
    }

}