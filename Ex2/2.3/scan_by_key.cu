#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <iostream>

template <typename Vector>
void print(const Vector& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << " ";
  std::cout << "\n";
}

thrust::device_vector<int> getValueVector(size_t N)
{
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 10);
    thrust::device_vector<int> values(N);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = dist(rng);
    }

    return values;
}

thrust::device_vector<int> getFlagVector(size_t N)
{
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<size_t> dist(1, 5);
    thrust::device_vector<int> flags(N);
    size_t currentSize = 0;
    while (currentSize < N)
    {
        if (N - currentSize < 5)
        {
            flags[currentSize] = 1;
            currentSize = flags.size();
        }
        else 
        {
            size_t segmentSize = dist(rng);
            flags[currentSize] = 1;
            currentSize += segmentSize;
        }
    }

    return flags;
}

struct value_flag_pair
{
    int value;
    int flag;
};

struct pair_binary_op
{
    __host__ __device__
    value_flag_pair operator()(int value, int flag) const
    {
        value_flag_pair pair;
        pair.value = value;
        pair.flag = flag;
        return pair;
    }
};   

struct scan_binary_op
{
    __host__ __device__
    value_flag_pair operator()(value_flag_pair const& first, value_flag_pair const& second) const
    {
        value_flag_pair pair;
        pair.value = first.value + second.value;
        pair.flag = 0;
        return pair;
    }
};

int main()
{
    size_t N = 20;
    thrust::device_vector<int> values = getValueVector(N);
    thrust::device_vector<int> flags = getFlagVector(N);

    thrust::device_vector<value_flag_pair> pairs(N);
    thrust::transform(values.begin(), values.end(), flags.begin(), pairs.begin(), pair_binary_op());

    thrust::device_vector<pairs> output(N);
    thrust::inclusive_scan(pairs.begin(), pairs.end(), output.begin(), scan_binary_op());

    for(size_t i = 0; i < N; i++)
        std::cout << output[i].value << " ";
    std::cout << "\n";
    return 0;
}