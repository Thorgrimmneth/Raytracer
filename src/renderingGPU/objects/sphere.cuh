#pragma once

#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "base_object.cuh"
struct Sphere{
    BaseObject base;
    float3 center1;
    float radius;
    float3 center2;
    int materialIndex;

    Sphere() = default;
    Sphere(float3 c, float r) : center1(c), radius(r), center2(c) {}
    Sphere(float3 c1, float3 c2, float r) : center1(c1), center2(c2), radius(r) {}
    __device__
    bool intersectGeometry(const Ray& ray, float& p_t1, float& p_t2) const;
    __device__
    bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;
    __device__
    bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const;
    __device__
    float3 computeNormal(const float3 & point, const double time) const;
};