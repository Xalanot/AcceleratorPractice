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
    thrust::uniform_int_distribution<int> dist(0, 100);
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
    thrust::uniform_int_distribution<size_t> dist(50, 100);
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
    flags[0] = 1;
    thrust::transform(keys.begin(), keys.end() - 1, keys.begin() + 1, flags.begin() + 1, thrust::not_equal_to<int>());
    return flags;
}

thrust::device_vector<int> generateKeys(thrust::device_vector<int> const& flags)
{
    thrust::device_vector<int> keys(flags.size());
    thrust::inclusive_scan(flags.begin() + 1, flags.end(), keys.begin() + 1);
    return keys;
}

int main(int argc, char** argv)
{
    int N = std::atoi(argv[1]);
    thrust::device_vector<int> values = getValueVector(N);
    thrust::device_vector<int> keys = getKeyVector(N);
    thrust::device_vector<int> flags = generateFlags(keys);
    thrust::device_vector<int> keys2 = generateKeys(flags);
    
    assert(keys == keys2);

    return 0;
}

