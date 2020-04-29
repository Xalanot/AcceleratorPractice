#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>


// This example shows how thrust::zip_iterator can be used to create a
// 'virtual' array of structures.  In this case the structure is a 3d
// vector type (Float3) whose (x,y,z) components will be stored in
// three separate float arrays.  The zip_iterator "zips" these arrays
// into a single virtual Float3 array.



// We'll use a 3-tuple to store our 3d vector type
typedef thrust::tuple<float,float,float> Float3;


// This functor implements the dot product between 3d vectors
struct DotProduct : public thrust::binary_function<Float3,Float3,float>
{
    __host__ __device__
        float operator()(const Float3& a, const Float3& b) const
        {
            return thrust::get<0>(a) * thrust::get<0>(b) +    // x components
                   thrust::get<1>(a) * thrust::get<1>(b) +    // y components
                   thrust::get<2>(a) * thrust::get<2>(b);     // z components
        }
};


int main(void)
{
    thrust::device_vector<float> A0(3);  // x components of the 'A' vectors
    thrust::device_vector<float> A1(3);  // y components of the 'A' vectors
    thrust::device_vector<float> A2(3));  // z components of the 'A' vectors
    A0[0] = 1; A1[0] = 4; A2[0] = 7;
    A0[1] = 2; A1[1] = 5; A2[1] = 8;
    A0[2] = 3; A1[2] = 6; A2[2] = 9;

    thrust::device_vector<float> B0(3);  // x components of the 'B' vectors
    thrust::device_vector<float> B1(3);  // y components of the 'B' vectors
    thrust::device_vector<float> B2(3));  // z components of the 'B' vectors
    B0[0] = 10; B1[0] = 13; B1[0] = 16;
    B0[1] = 11; B1[1] = 14; B1[1] = 17;
    B0[2] = 12; B1[2] = 15; B1[2] = 18;

    // Storage for result of each dot product
    Float3 init(0, 0, 0);

    Float3 result = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(A0.begin(), A1.begin(), A2.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(A0.end(), A1.end(), A2.end())),
                                   thrust::make_zip_iterator(thrust::make_tuple(B0.begin(), B1.begin(), B2.begin())),
                                   init;
                                   thrust::plus<float>());

                       
    std::cout << "result: ";
    std::cout << "(" << thrust::get<0>(result) << "," << thrust::get<1>(result) << "," << thrust::get<2>(result) << ")";
    std::cout << "\n";

    return 0;
}

