#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

void printHelp()
{
    std::cout << "Please provide the size as argument" << std::endl;
}

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(float const& x, float const& y) const
    {
        return a * x + y;
    }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  thrust::device_vector<float> temp(X.size());
   
  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);
    
  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(int argc, char** argv)
{
    if (argc != 2) printHelp();

    int N = atoi(argv[1]);

    thrust::host_vector<float> X_h(N, 1);
    thrust::host_vector<float> Y_h(N);
    thrust::sequence(Y_h.begin(), Y_h.end());

    thrust::device_vector<float> X_d(X_h);
    thrust::device_vector<float> Y_d(Y_h);

    saxpy_fast(3, X_d, Y_d);

    return 0;
}