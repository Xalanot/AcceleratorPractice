#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>

// this example fuses a gather operation with a reduction for
// greater efficiency than separate gather() and reduce() calls

int main(void)
{
    // gather locations
    thrust::device_vector<int> map(4);
    map[0] = 3;
    map[1] = 1;
    map[2] = 0;
    map[3] = 5;

    // array to gather from
    thrust::device_vector<int> source(6);
    source[0] = 10;
    source[1] = 20;
    source[2] = 30;
    source[3] = 40;
    source[4] = 50;
    source[5] = 60;

    thrust::device_vector<int> result(4);

    // source[map[i]] = source[map[i]] + source[map[i+1]]
    thrust::transform(
        thrust::make_permutation_iterator(source.begin(), map.begin()),
        thrust::make_permutation_iterator(source.begin(), map.end()),
        thrust::make_permutation_iterator(source.begin(), map.begin()+1),
        result.begin(),
        thrust::plus<int>());

    // print sum
    /*thrust::copy(thrust::make_permutation_iterator(source.begin(), map.begin()),
                 thrust::make_permutation_iterator(source.begin(), map.end()), 
                 std::ostream_iterator<int>(std::cout, "\t")); */
    thrust::copy(result.begin(),
                 result.end(), 
                 std::ostream_iterator<int>(std::cout, "\t"));              

    return 0;
}
