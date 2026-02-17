#pragma once

#include "cuda_op.cuh"
#include "cuda_ray.cuh"

struct AABB{
    float4 min;
    float4 max;

    __device__
    inline float3 centroid();

    __device__
    float area();

    __device__
    bool intersect( const Ray & ray, const float p_tMin, const float p_tMax ) const;
};