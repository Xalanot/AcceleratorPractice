#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
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

int main()
{
    thrust::device_vector<int> vec_d(3, 1);
    thrust::device_vector<int> vec_d2(4, 2);
    std::vector<thrust::device_vector<int>> vectors{vec_d, vec_d2};

    auto concatVec = concatInSingleVector<int>(vectors);

    int size = concatVec.size();

    std::cout << "device size: " << size << std::endl;

    thrust::host_vector<int> vec_h = concatVec;

    for (auto const& ele : vec_h)
    {
        std::cout << ele << std::endl;
    }

    std::cout << "size: " << vec_h.size() << std::endl;

    return 0;*/
}