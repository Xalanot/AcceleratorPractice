#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

// This example shows how to compute a bounding box
// for a set of points in two dimensions.

struct point2d
{
  float x, y;
  
  __host__ __device__
  point2d() : x(0), y(0) {}
  
  __host__ __device__
  point2d(float _x, float _y) : x(_x), y(_y) {}
};

// bounding box type
struct bbox
{
  // construct an empty box
  __host__ __device__
  bbox() {}

  // construct a box from a single point
  __host__ __device__
  bbox(const point2d &point)
    : lower_left(point), upper_right(point)
  {}

  // construct a box from a single point
  __host__ __device__
  bbox& operator=(const point2d &point)
  {
    lower_left = point;
    upper_right = point;
    return *this;
  }

  // construct a box from a pair of points
  __host__ __device__
  bbox(const point2d &ll, const point2d &ur)
    : lower_left(ll), upper_right(ur)
  {}

  point2d lower_left, upper_right;
};

// reduce a pair of bounding boxes (a,b) to a bounding box containing intersection of a and b
struct bbox_reduction : public thrust::binary_function<bbox,bbox,bbox>
{
  __host__ __device__
  bbox operator()(bbox a, bbox b)
  {
    // lower left corner
    point2d ll(thrust::max(a.lower_left.x, b.lower_left.x), thrust::max(a.lower_left.y, b.lower_left.y));
    
    // upper right corner
    point2d ur(thrust::min(a.upper_right.x, b.upper_right.x), thrust::min(a.upper_right.y, b.upper_right.y));
    
    return bbox(ll, ur);
  }
};

int main(void)
{
  const size_t N = 40;
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
  
  // allocate storage for points
  thrust::device_vector<bbox> bboxes(N);
  
  // generate some random points in the unit square
  for(size_t i = 0; i < N; i++)
  {
      float x = u01(rng);
      float y = u01(rng);
      point2d ll(x,y);

      x = u01(rng);
      y = u01(rng);
      point2d ur(x,y);

      bboxes[i] = bbox(ll, ur);
  }
  
  // binary reduction operation
  bbox_reduction binary_op;
  
  // compute the intersection bounding box for the point set
  bbox result = thrust::reduce(bboxes.begin(), bboxes.end(), bboxes[0], binary_op);
  
  // print output
  std::cout << "intersection bounding box " << std::fixed;
  std::cout << "(" << result.lower_left.x  << "," << result.lower_left.y  << ") ";
  std::cout << "(" << result.upper_right.x << "," << result.upper_right.y << ")" << std::endl;
  
  return 0;
}
