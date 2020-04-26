#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <vector>

/*template<typename T>
size_t concatInSingleVector(thrust::device_vector<thrust::device_vector<T>> const& vectors)
{
    // calculate final size
    size_t size = 0;
    for (size_t i = 0; i < vectors.size(); ++i)
    {
        size += vectors[i].size();
    }

    return size;

    /*
    thrust::device_vector<T> returnVec;
    returnVec.reserve(size);

    size_t offset = 0;
    for (size_t i = 0; i < vectors.size(); ++i)
    {
        thrust::copy(vectors[i].begin(), vectors[i].end(), returnVec.begin() + offset);
        offset += vectors[i].size();
    }

    return returnVec;
}*/

int main()
{
    thrust::device_vector<int> vec_d(3, 1);
    thrust::device_vector<int> vec_d2(4, 2);
    std::vector<thrust::device_vector<int>> vectors{vec_d, vec_d2};
    /*vectors[0] = vec_d;
    vectors[1] = vec_d2;

    auto concatVec = concatInSingleVector<int>(vectors);

    std::cout << concatVec << std::endl;

    thrust::host_vector<int> vec_h = concatVec;

    for (auto const& ele : vec_h)
    {
        std::cout << ele << std::endl;
    }

    return 0;*/
}