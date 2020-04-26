#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 5);
  return dist(rng);
}

int main(void)
{
  // generate random data on the host
  thrust::host_vector<int> h_vec(100);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;

  // initial value of the reduction
  int init = 1; 
 
  // binary operation used to reduce values
  thrust::multiplies<int> binary_op;

  // compute mul on the device
  int mul = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);

  // print the mul
  std::cout << "mul is " << mul << std::endl;

  return 0;
}
