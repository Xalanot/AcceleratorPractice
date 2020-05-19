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

int main()
{
    size_t N = 20;
    thrust::device_vector<int> values = getValueVector(N);
    thrust::device_vector<int> flags = getFlagVector(N);

    print(flags);
    return 0;
}