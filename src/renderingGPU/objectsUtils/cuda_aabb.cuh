#pragma once

#include "../utils/cuda_op.cuh"
#include "../raytracingUtils/cuda_ray.cuh"

struct AABB{
    float3 min;
    float3 max;

    __device__
    inline float3 centroid();

    __device__
    float area();

    __device__
    bool intersect( const Ray & ray, const float p_tMin, const float p_tMax ) const;
};