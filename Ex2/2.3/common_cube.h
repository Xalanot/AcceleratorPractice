#include <thrust/device_vector.h>

// define a 2d float vector
typedef thrust::tuple<float,float,float> vec3;

thrust::device_vector<vec3> createCube(int xOffset, int yOffset, int zOffset)
{
    thrust::device_vector<vec3> cube(24);

    cube[0] = vec3(0 + xOffset, 0 + yOffset, 0 + zOffset);  // First Quad Left
    cube[1] = vec3(0 + xOffset, 0 + yOffset , 1 + zOffset);
    cube[2] = vec3(0 + xOffset, 1 + yOffset, 0 + zOffset);
    cube[3] = vec3(0 + xOffset, 1 + yOffset, 1 + zOffset);  
    cube[4] = vec3(0 + xOffset, 0 + yOffset, 0 + zOffset); // Second Quad Front
    cube[5] = vec3(0 + xOffset, 0 + yOffset, 1 + zOffset);
    cube[6] = vec3(1 + xOffset, 0 + yOffset, 0 + zOffset);  
    cube[7] = vec3(1 + xOffset, 0 + yOffset, 1 + zOffset);
    cube[8] = vec3(1 + xOffset, 0 + yOffset, 0 + zOffset); // Third Quad Right
    cube[9] = vec3(1 + xOffset, 0 + yOffset, 1 + zOffset);
    cube[10] = vec3(1 + xOffset, 1 + yOffset, 0 + zOffset);
    cube[11] = vec3(1 + xOffset, 1 + yOffset, 1 + zOffset);
    cube[12] = vec3(0 + xOffset, 1 + yOffset, 0 + zOffset); // Forth Quad Back
    cube[13] = vec3(0 + xOffset, 1 + yOffset, 1 + zOffset);
    cube[14] = vec3(1 + xOffset, 1 + yOffset, 0 + zOffset);
    cube[15] = vec3(1 + xOffset, 1 + yOffset, 1 + zOffset);
    cube[16] = vec3(0 + xOffset, 0 + yOffset, 0 + zOffset); // Fith Quad Bottom
    cube[17] = vec3(0 + xOffset, 1 + yOffset, 0 + zOffset);
    cube[18] = vec3(1 + xOffset, 0 + yOffset, 0+ zOffset);
    cube[19] = vec3(1 + xOffset, 1 + yOffset, 0 + zOffset);
    cube[20] = vec3(0 + xOffset, 0 + yOffset, 1 + zOffset); // Sixth Quad Top
    cube[21] = vec3(0 + xOffset, 1 + yOffset, 1 + zOffset);
    cube[22] = vec3(1 + xOffset, 0 + yOffset, 1 + zOffset);
    cube[23] = vec3(1 + xOffset, 1 + yOffset, 1 + zOffset);
}