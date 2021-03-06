#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <iostream>

template <typename HeadFlagType>
struct head_flag_predicate 
    : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
    __host__ __device__
    bool operator()(HeadFlagType, HeadFlagType right) const
    {
        return !right;
    }
};

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
    thrust::uniform_int_distribution<size_t> dist(50, 100);
    thrust::device_vector<int> flags(N);
    size_t currentSize = 0;
    while (currentSize < N)
    {
        if (N - currentSize < 100)
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

thrust::device_vector<int> getValueVectorFromPair(thrust::device_vector<value_flag_pair> const& pairs)
{
    thrust::host_vector<value_flag_pair> h_pairs(pairs);
    thrust::device_vector<int> values(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        values[i] = h_pairs[i].value;
    }

    return values;
}

std::ostream &operator<<(std::ostream &os, const value_flag_pair &pair)
{
  os << pair.value;
  return os;
}

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
        if (second.flag)
        {
            pair.value = second.value;
            pair.flag = 1;
        }
        else if (first.flag)
        {
            pair.value = first.value + second.value;
            pair.flag = 1;
        }
        else 
        {
            pair.value = first.value + second.value;
            pair.flag = 0;
        }
        return pair;
    }
};

int main(int argc, char** argv)
{
    size_t N = std::stoi(argv[1]);;
    thrust::device_vector<int> values = getValueVector(N);
    thrust::device_vector<int> flags = getFlagVector(N);

    thrust::device_vector<value_flag_pair> pairs(N);
    thrust::transform(values.begin(), values.end(), flags.begin(), pairs.begin(), pair_binary_op());

    thrust::device_vector<value_flag_pair> output(N);
    thrust::inclusive_scan(pairs.begin(), pairs.end(), output.begin(), scan_binary_op());

    thrust::device_vector<int> output2(N);
    thrust::inclusive_scan_by_key(flags.begin(), flags.end(), values.begin(), output2.begin(), head_flag_predicate<int>());
    assert(getValueVectorFromPair(output) == output2);
    return 0;
}