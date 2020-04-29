#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/random.h>

// This example generates random points in the 
// unit square [0,1)x[0,1) and then removes all 
// points where x^2 + y^2 > 1
//
// The x and y coordinates are stored in separate arrays
// and a zip_iterator is used to combine them together

template <typename T>
struct is_inside_circle
{
    template <typename Tuple>
    inline __host__ __device__
    bool operator()(const Tuple& tuple) const
    {
        // unpack the tuple into x and y coordinates
        const T x = thrust::get<0>(tuple);
        const T y = thrust::get<1>(tuple);

        if (x*x + y*y > 1)
            return false;
        else
            return true;
    }
};

int main(void)
{
    const size_t N = 20;

    // generate random points in the unit square on the host
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    thrust::host_vector<float> x(N);
    thrust::host_vector<float> y(N);
    for(size_t i = 0; i < N; i++)
    {
        x[i] = u01(rng);
        y[i] = u01(rng);
    }

    // print the initial points
    std::cout << std::fixed;
    std::cout << "Generated " << N << " points" << std::endl;
    for(size_t i = 0; i < N; i++)
        std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;
    std::cout << std::endl;

    // remove points where x^2 + y^2 > 1 and determine new array sizes
    hrust::partition(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
                     is_inside_circle<float>());

    // print the partioned points
    for(size_t i = 0; i < x.size(); i++)
        std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;

    return 0;
}

