#pragma once

#include "cuda_aabb.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"

struct BVH{
    AABB bbox;
    int left = -1;
    int right = -1;
    int firstTriangleIndex;
    int lastTriangleIndex;

    __device__
    inline bool isLeaf() const { return ( left == -1); }
};