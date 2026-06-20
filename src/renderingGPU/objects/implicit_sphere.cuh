#pragma once

#include "../utils/objectType.h"
#include "../../../devicePrograms/launch_params.cuh"

struct ImplicitSphere
{
    float4 center1; // xyz = center, w = radius
    float4 center2; // xyz = center2, w = materialIndex

    ImplicitSphere(float3 c1, float r, float3 c2, int materialIndex)
        : center1(make_float4(c1, r)), center2(make_float4(c2, intBitsToFloat(materialIndex)))
    {
    }

    HD_FORCEINLINE 
    float3 getCenter1() const { return make_float3(center1); }

    HD_FORCEINLINE
    float getRadius() const { return center1.w; }

    HD_FORCEINLINE
    float3 getCenter2() const { return make_float3(center2); }

    HD_FORCEINLINE
    int getMaterialIndex() const { return floatBitsToInt(center2.w); }

    HOST
    void setMaterialIndex(int materialIndex)
    {
        center2.w = intBitsToFloat(materialIndex);
    }

    D_FORCEINLINE 
    float sdf(const float3 &point, const double time = 0) const
    {
        float4 center = center1 + (center2 - center1) * time;
        return length(point - make_float3(center)) - getRadius();
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
    float3 computeNormal(const float3 &point, const double time = 0) const
    {
        float3 center = getCenter1() + (getCenter2() - getCenter1()) * time;
        return normalize(point - center);
    }

   /*D_FORCEINLINE 
    bool intersect(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax,
                                              OptixHit &p_hitRecord) const
    {
        return false;
        float t = p_tMin;
        const float threshold = 1e-4f;
        const float minStep = 1e-4f;

        for (int i = 0; i < 48; i++)
        {
            if (t >= p_tMax)
                return false;

            float3 point = origin + direction * t;
            float dist = sdf(point);

            if (fabs(dist) < threshold)
            {
                // Intersection found, fill p_hitRecord.
                p_hitRecord.setHitInfo(point, computeNormal(point), t, getMaterialIndex(), 0, HIT_SPHERE_IMPLICIT);
                p_hitRecord.faceNormal(direction);
                return true;
            }
            t += max(fabs(dist), minStep);
        }
        return false;
    }

    D_FORCEINLINE 
    bool intersectAny(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax) const
    {
        float t = p_tMin + 1e-3f;
        const float threshold = 1e-4f;
        const float minStep = 1e-4f;

        for (int i = 0; i < 48; ++i)
        {
            if (t >= p_tMax)
                return false;

            float3 p = origin + direction * t;

            float dist = sdf(p);

            if (fabs(dist) < threshold)
                return true;

            t += max(fabs(dist), minStep);
        }

        return false;
    }*/
};