#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <string>

// this functor clamps a value to the range [lo, hi]
template <typename T>
struct negate_and_clamp : public thrust::unary_function<T,T>
{
    T lo, hi;

    __host__ __device__
    negate_and_clamp(T _lo, T _hi) : lo(_lo), hi(_hi) {}

    __host__ __device__
    T operator()(T x)
    {
        // first negate
        x = -x;

        // now clamp
        if (x < lo)
            return lo;
        else if (x < hi)
            return x;
        else
            return hi;
    }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
    std::cout << "\n";
}


int main(void)
{
    // clamp values to the range [1, 5]
    int lo = 1;
    int hi = 5;

    // initialize values
    thrust::device_vector<int> values(8);
    thrust::sequence(values.begin(), values.end(), -9);
    
    print_range("values         ", values.begin(), values.end());

    // create a transform_iterator that applies clamp() to the values array
    auto cv_begin = thrust::make_transform_iterator(values.begin(), negate_and_clamp<int>(lo, hi));
    auto cv_end   = cv_begin + values.size();
    
    // now [clamped_begin, clamped_end) defines a sequence of clamped values
    print_range("clamped values ", cv_begin, cv_end);

    return 0;
}

