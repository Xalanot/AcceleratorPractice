#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>

// power<T> computes the power of a number f_n(x) -> x^n
template <typename T>
struct n_power
{
    n_power(int _n) : n(_n)

    int n;

    __host__ __device__
        T operator()(const T& x) const { 
            return static_cast<T>(pow(x, n));
        }
};

int main(void)
{
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

    // setup arguments
    int n = 2;
    n_power<float> unary_op(n);
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    double norm = pow(( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) ), 1 / n);

    std::cout << "norm is " << norm << std::endl;

    return 0;
}

