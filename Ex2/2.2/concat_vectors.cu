#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

template<typename T>
thrust::device_vector<T> concatInSingleVector(thrust::device_vector<T> const& vectors)
{
    // calculate final size
    size_t size = 0;
    for (auto const& vec : vectors)
    {
        size += vec.size();
    }

    thrust::device_vector<T> returnVec;
    returnVec.reserve(size);

    size_t offset = 0;
    for (auto const& vec: vectors)
    {
        thrust::copy(vec.begin(), vec.end(), returnVec.begin() + offset);
        offset += vec.size();
    }

    return returnVec;
}

int main()
{
    thrust::device_vector vec_d {1, 2, 3};
    thrust::device_vector vec_d2 {4, 5, 6};
    thrust::device_vector vectors = {vec_d, vec_d2};

    auto concatVec = concatInSingleVector(vectors);

    thrust::host_vector vec_h = concatVec;

    for (auto const& ele : vec_h)
    {
        std::cout << ele << std::endl;
    }

    return 0;
}