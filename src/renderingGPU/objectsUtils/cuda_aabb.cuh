#pragma once

#include "../utils/cuda_op.cuh"
#include "../raytracingUtils/cuda_ray.cuh"

struct AABB{
    float4 min = float4f(+INFINITY);
    float4 max = float4f(-INFINITY);

    __host__ __device__
    float4 centroid() const;

    __host__ __device__
    float area();

    __device__
    bool intersect( const Ray & ray, const float p_tMin, const float p_tMax ) const;

    __host__
    void extend(const AABB& a);

    __host__
    void extend(const float3& a);

    __host__
    void extend(const float4& a);
};