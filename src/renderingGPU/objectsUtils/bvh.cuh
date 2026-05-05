#pragma once

#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"

struct BVH{
    AABB bbox;
    int left = -1;
    int right = -1;
    int firstTriangleIndex;
    int lastTriangleIndex;

    __device__
    inline bool isLeaf() const { return ( left == -1); }
};