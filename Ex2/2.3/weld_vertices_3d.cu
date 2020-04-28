#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <iostream>

#include "common_cube.h"

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

// define a 3d float vector
typedef thrust::tuple<float,float,float> vec3;

// defince a face vector
typedef thrust::tuple<unsigned int, unsigned int, unsigned int, unsigned int> faceVec;

int main(void)
{
    // allocate memory for input mesh representation
    thrust::device_vector<vec3> input(192); 
    std::vector<thrust::device_vector<vec3>> inputs(8);
    inputs[0] = createCube(0, 0, 0);
    inputs[1] = createCube(-1, 0, 0);
    inputs[2] = createCube(0, -1, 0);
    inputs[3] = createCube(0, 0, -1);
    inputs[4] = createCube(-1, -1, 0);
    inputs[5] = createCube(-1, 0, -1);
    inputs[6] = createCube(0, -1, -1);
    inputs[7] = createCube(-1, -1, -1);
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        thrust::copy(inputs[i].begin(), inputs[i].end(), input.begin() + i * 24);
    }

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
        if (i % 4 == 0)
        {
            std::cout << "new quad" << std::endl;
        }
        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
    }

    thrust::device_vector<faceVec> faces(indices.size() / 4);
    for (size_t i = 0; i < indices.size(); i+= 4)
    {
        faceVec face(indices[i], indices[i+1], indices[i+2], indices[i+3]);
        faces[i / 4] = face;
    }

    std::cout << "faces" << std::endl;
    for (size_t i = 0; i < faces.size(); ++i)
    {
        faceVec face = faces[i];
        std::cout << " faces[" << i << "] = (" << thrust::get<0>(face) << "," << thrust::get<1>(face) << "," << thrust::get<2>(face) << "," << thrust::get<3>(face) << ")" << std::endl;
    }

    return 0;
}

