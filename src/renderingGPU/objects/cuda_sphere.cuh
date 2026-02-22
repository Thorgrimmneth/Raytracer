#pragma once

#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"

struct Sphere{
    float3 center1;
    float3 center2;
    float radius;

    int materialIndex;

    __device__
    bool intersectGeometry(const Ray& ray, float& p_t1, float& p_t2) const;
    __device__
    bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;
    __device__
    bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const;
    __device__
    float3 computeNormal(const float3 & point, const double time) const;
};