#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>

// BinaryPredicate for the head flag segment representation
// equivalent to thrust::not2(thrust::project2nd<int,int>()));
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
    thrust::uniform_real_distribution<int> dist(0, 100);
    thrust::device_vector<int> values(N);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = dist(rng);
    }

    return values;
}

thrust::device_vector<int> getKeyVector(size_t N)
{
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<size_t> dist(1, 5);
    thrust::device_vector<int> keys(N);
    size_t currentSize = 0;
    int currentKey = 0;
    while (currentSize < N)
    {
        if (N - currentSize < 100)
        {
            for (size_t i = currentSize; i < keys.size(); ++i)
            {
                keys[i] = currentKey;
                currentSize = keys.size();
            }
        }
        else 
        {
            size_t keySize = dist(rng);
            for (size_t i = currentSize; i < currentSize + keySize; ++i)
            {
                keys[i] = currentKey;
            }
            currentSize += keySize;
            currentKey++;
        }
    }

    return keys;
}

thrust::device_vector<int> generateFlags(thrust::device_vector<int> const& keys)
{
    thrust::device_vector<int> flags(keys.size());
    thrust::transform(keys.begin(), keys.end(), flags.begin(), flags.begin(), thrust::not_equal_to<int>());
    return flags;
}

int main(void)
{
    int N = 20;
    thrust::device_vector<int> values = getValueVector(N);
    thrust::device_vector<int> keys = getKeyVector(N);
    thrust::device_vector<int> flags = generateFlags(keys);
    
    print(keys);
    print(flags);

    return 0;
}

