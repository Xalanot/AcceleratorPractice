#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <iostream>

/*
 * This example "welds" triangle vertices together by taking as
 * input "triangle soup" and eliminating redundant vertex positions
 * and shared edges.  A connected mesh is the result.
 * 
 *
 * Input: 9 vertices representing a mesh with 3 triangles
 *  
 *  Mesh              Vertices 
 *    ------           (2)      (5)--(4)    (8)      
 *    | \ 2| \          | \       \   |      | \
 *    |  \ |  \   <->   |  \       \  |      |  \
 *    | 0 \| 1 \        |   \       \ |      |   \
 *    -----------      (0)--(1)      (3)    (6)--(7)
 *
 *   (vertex 1 equals vertex 3, vertex 2 equals vertex 5, ...)
 *
 * Output: mesh representation with 5 vertices and 9 indices
 *
 *  Vertices            Indices
 *   (1)--(3)            [(0,2,1),
 *    | \  | \            (2,3,1), 
 *    |  \ |  \           (2,4,3)]
 *    |   \|   \
 *   (0)--(2)--(4)
 */

// define a 2d float vector
typedef thrust::tuple<float,float,float> vec3;

int main(void)
{
    // allocate memory for input mesh representation
    thrust::device_vector<vec3> input(24);

    input[0] = vec3(0,0,0);  // First Quad Left
    input[1] = vec3(0,0,1);
    input[2] = vec3(0,1,0);
    input[3] = vec3(0,1,1);  
    input[4] = vec3(0,0,0); // Second Quad Front
    input[5] = vec3(0,0,1);
    input[6] = vec3(1,0,0);  
    input[7] = vec3(1,0,1);
    input[8] = vec3(1,0,0); // Third Quad Right
    input[9] = vec3(1,0,1);
    input[10] = vec3(1,1,0);
    input[11] = vec3(1,1,1);
    input[12] = vec3(0,1,0); // Forth Quad Back
    input[13] = vec3(0,1,1);
    input[14] = vec3(1,1,0);
    input[15] = vec3(1,1,1);
    input[16] = vec3(0,0,0); // Fith Quad Bottom
    input[17] = vec3(0,1,0);
    input[18] = vec3(1,0,0);
    input[19] = vec3(1,1,0);
    input[20] = vec3(0,0,1); // Sixth Quad Top
    input[21] = vec3(0,1,1);
    input[22] = vec3(1,0,1);
    input[23] = vec3(1,1,1);

    // allocate space for output mesh representation
    thrust::device_vector<vec3> vertices = input;
    thrust::device_vector<unsigned int> indices(input.size());

    // sort vertices to bring duplicates together
    thrust::sort(vertices.begin(), vertices.end());

    // find unique vertices and erase redundancies
    vertices.erase(thrust::unique(vertices.begin(), vertices.end()), vertices.end());

    // find index of each input vertex in the list of unique vertices
    thrust::lower_bound(vertices.begin(), vertices.end(),
                        input.begin(), input.end(),
                        indices.begin());

    // print output mesh representation
    std::cout << "Output Representation" << std::endl;
    for(size_t i = 0; i < vertices.size(); i++)
    {
        vec3 v = vertices[i];
        std::cout << " vertices[" << i << "] = (" << thrust::get<0>(v) << "," << thrust::get<1>(v) << "," << thrust::get<2>(v) << ")" << std::endl;
    }
    for(size_t i = 0; i < indices.size(); i++)
    {
        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
    }

    return 0;
}

