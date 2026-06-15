#pragma once

#include "../../../devicePrograms/launch_params.cuh"
struct Sphere
{
    float4 center1; // xyz = center, w = radius
    float4 center2; // xyz = center2, w = materialIndex

    Sphere() = default;
    Sphere(float3 c, float r) : center1(make_float4(c, r)), center2(make_float4(c, 0)) {}
    Sphere(float3 c, float r, int matIndex) : center1(make_float4(c, r)), center2(make_float4(c, intBitsToFloat(matIndex))) {}
    Sphere(float3 c1, float3 c2, float r) : center1(make_float4(c1, r)), center2(make_float4(c2, 0)) {}

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
    bool intersect(const float3 &origin, const float3 &direction, const float tMin, const float tMax, OptixHit &hit) const
    {
        const float3 center = getCenter1() + 0.f * (getCenter2() - getCenter1()); // 0.f = time

        const float3 oc = origin - center;
        const float half_b = dot(direction, oc);
        const float c = dot(oc, oc) - getRadius() * getRadius();

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

        const float3 p = origin + t * direction;
        const float3 n = normalize(p - center);
        hit.setHitInfo(p, n, t, getMaterialIndex(), 0, HIT_SPHERE);
        hit.faceNormal(direction);
        return true;
    }
    
    D_FORCEINLINE 
    bool intersectAny(const float3 &origin, const float3 &direction, const float tMin, const float tMax) const
    {
        const float3 center = getCenter1() + 0.f * (getCenter2() - getCenter1()); // 0.f = time

        const float3 oc = origin - center;
        const float half_b = dot(direction, oc);
        const float c = dot(oc, oc) - getRadius() * getRadius();

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
        float3 c0 = getCenter1();
        float3 c1 = getCenter2();

        float3 center = c0 + time * (c1 - c0);

        return normalize(point - center);
    }
};