#pragma once

#include "../raytracingUtils/hitrecord.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../utils/macro.cuh"

struct ImplicitSphere
{
    float3 center1;
    float radius;
    float3 center2;
    int materialIndex;

    D_FORCEINLINE 
    float sdf(const float3 &point, const double time = 0) const
    {
        float3 center = center1 + (center2 - center1) * time;
        return length(point - center) - radius;
    }

    /*
    D_FORCEINLINE
    float3 ImplicitSphere::computeNormal(const float3 &point, const double time) const
    {
        float e = 1.e-4f;
        return normalize(make_float3(1.f, -1.f, -1.f) * sdf(point + make_float3(e, -e, -e))
                       + make_float3(-1.f, -1.f, 1.f) * sdf(point + make_float3(-e, -e, e))
                       + make_float3(-1.f, 1.f, -1.f) * sdf(point + make_float3(-e, e, -e))
                       + make_float3(1.f ,1.f, 1.f) * sdf(point + make_float3(e, e, e)));
    }*/

    D_FORCEINLINE 
    float3 computeNormal(const float3 &point, const double time) const
    {
        float3 center = center1 + (center2 - center1) * time;
        return normalize(point - center);
    }

    D_FORCEINLINE 
    bool intersect(const Ray &p_ray, const float p_tMin, const float p_tMax,
                                              HitRecord &p_hitRecord) const
    {
        float t = p_tMin;
        const float threshold = 1e-4f;
        const float minStep = 1e-4f;

        for (int i = 0; i < 48; i++)
        {
            if (t >= p_tMax)
                return false;

            float3 point = p_ray.pointAtT(t);
            float dist = sdf(point, p_ray.time);

            if (fabs(dist) < threshold)
            {
                // Intersection found, fill p_hitRecord.
                p_hitRecord.point = point;
                p_hitRecord.normal = computeNormal(point, p_ray.time);
                p_hitRecord.faceNormal(p_ray.direction);
                p_hitRecord.distance = t;
                p_hitRecord.materialIndex = materialIndex;
                return true;
            }
            t += max(fabs(dist), minStep);
        }
        return false;
    }

    D_FORCEINLINE 
    bool intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
    {
        float t = p_tMin + 1e-3f;
        const float threshold = 1e-4f;
        const float minStep = 1e-4f;

        for (int i = 0; i < 48; ++i)
        {
            if (t >= p_tMax)
                return false;

            float3 p = p_ray.origin + p_ray.direction * t;

            float dist = sdf(p, p_ray.time);

            if (fabs(dist) < threshold)
                return true;

            t += max(fabs(dist), minStep);
        }

        return false;
    }
};