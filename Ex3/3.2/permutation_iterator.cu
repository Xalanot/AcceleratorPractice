#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>

// this example fuses a gather operation with a reduction for
// greater efficiency than separate gather() and reduce() calls

struct add_neighbours : public thrust::unary_function<int,int>
{
    add_neighbours(thrust::device_vector<int> source) : source(source) {}
    thrust::device_vector<int> source;

    __host__ __device__
    int operator()(int x) 
    { 
        return source[x] + source[x+1];
    }
};

int main(void)
{
    // gather locations
    thrust::device_vector<int> map(4);
    map[0] = 3;
    map[1] = 1;
    map[2] = 0;
    map[3] = 4;

    // array to gather from
    thrust::device_vector<int> source(6);
    source[0] = 10;
    source[1] = 20;
    source[2] = 30;
    source[3] = 40;
    source[4] = 50;
    source[5] = 60;

    // fuse gather with reduction: 
    //   sum = source[map[0]] + source[map[1]] + ...
    thrust::transform(map.begin(), map.end(),
                      thrust::make_permutation_iterator(source.begin(), map.begin()),
                      add_neighbours(source));

    // print sum
    thrust::copy(thrust::make_permutation_iterator(source.begin(), map.begin(), thrust::make_permutation_iterator(source.begin(), map.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}
