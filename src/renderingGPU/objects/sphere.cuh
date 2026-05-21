#pragma once

#include "../utils/objects.cuh"

struct Sphere
{
    float3 center1;
    float radius;
    float3 center2;
    int materialIndex;

    Sphere() = default;
    Sphere(float3 c, float r) : center1(c), radius(r), center2(c) {}
    Sphere(float3 c1, float3 c2, float r) : center1(c1), center2(c2), radius(r) {}

    D_FORCEINLINE 
    bool intersect(const Ray &ray, const float tMin, const float tMax, HitRecord &hit) const
    {
        const float3 center = center1 + ray.time * (center2 - center1);

        const float3 oc = ray.origin - center;
        const float half_b = dot(ray.direction, oc);
        const float c = dot(oc, oc) - radius * radius;

        const float delta = half_b * half_b - c;
        if (delta < 0.f)
            return false;

        const float sqrtDelta = sqrtf(delta);

        float t = -half_b - sqrtDelta;
        if (t < tMin || t > tMax)
        {
            t = -half_b + sqrtDelta;
            if (t < tMin || t > tMax)
                return false;
        }

        const float3 p = ray.origin + t * ray.direction;
        const float3 n = normalize(p - center);

        hit.point = p;
        hit.normal = n;
        hit.faceNormal(ray.direction);
        hit.distance = t;
        hit.materialIndex = materialIndex;

        return true;
    }
    
    D_FORCEINLINE 
    bool intersectAny(const Ray &ray, const float tMin, const float tMax) const
    {
        const float3 center = center1 + ray.time * (center2 - center1);

        const float3 oc = ray.origin - center;
        const float half_b = dot(ray.direction, oc);
        const float c = dot(oc, oc) - radius * radius;

        const float delta = half_b * half_b - c;
        if (delta < 0.0f)
            return false;

        const float sqrtDelta = sqrtf(delta);

        float t = -half_b - sqrtDelta;

        if (t < tMin)
            t = -half_b + sqrtDelta;

        return t >= tMin && t <= tMax;
    }

    D_FORCEINLINE 
    float3 computeNormal(const float3 &point, const double time) const
    {
        float3 c0 = center1;
        float3 c1 = center2;

        float3 center = c0 + time * (c1 - c0);

        return normalize(point - center);
    }
};