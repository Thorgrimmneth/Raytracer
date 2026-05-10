#pragma once

#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"

struct ImplicitSphere
{
    float3 center1;
    float radius;
    float3 center2;
    int materialIndex;

    __host__ __device__
    float sdf(const float3 &point, const double time) const;

    __device__
    float3 computeNormal(const float3 &point, const double time) const;

    __device__
    bool intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const;

    __device__
    bool intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const;
};