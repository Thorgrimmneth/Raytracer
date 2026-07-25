#pragma once

#include "../../utils/rng_cpu.hpp"
#include "../utils/rng.cuh"
#include "../utils/op.cuh"
#include <optix.h>
#include <optix_stubs.h>

struct SphereAnalytic
{
    float radius;
    int materialIndex;

    static SphereAnalytic createRandomSphere(int materialIndex)
    {
        float radius = randomFloat() * 2.f + 0.2f;
        return create(radius, materialIndex);
    }

    static SphereAnalytic create(float r, int m) { return {r, m}; }

    H_INLINE OptixAabb computeWorldAABB(const float3 &translation) const
    {
        OptixAabb aabb;
        aabb.minX = translation.x - radius;
        aabb.minY = translation.y - radius;
        aabb.minZ = translation.z - radius;
        aabb.maxX = translation.x + radius;
        aabb.maxY = translation.y + radius;
        aabb.maxZ = translation.z + radius;
        return aabb;
    }

    D_FORCEINLINE void sampleSurfacePoint(float3 &p_point, float3 &p_normal, RNG &rng) const
    {
        float u = rng.nextFloat();
        float v = rng.nextFloat();

        float theta = 2.0f * M_PIf * u;
        float phi = acosf(2.0f * v - 1.0f);

        p_normal.x = sinf(phi) * cosf(theta);
        p_normal.y = sinf(phi) * sinf(theta);
        p_normal.z = cosf(phi);

        p_point = p_normal * radius;
    }

    HD_FORCEINLINE OptixAabb computeAABB() const
    {
        OptixAabb aabb;
        aabb.minX = -radius;
        aabb.minY = -radius;
        aabb.minZ = -radius;
        aabb.maxX = radius;
        aabb.maxY = radius;
        aabb.maxZ = radius;
        return aabb;
    }

    D_FORCEINLINE float3 getNormal(const float3 &point, const float3 &center) const
    {
        float3 localPoint = point - center;
        float l = length(localPoint);
        float4 normal = make_float4(l - radius, localPoint / l);
        return make_float3(normal.y, normal.z, normal.w);
    }

    __device__ float intersect(const float3 &center, const float3 &ray_origin, const float3 &ray_direction, float &tMin, const float tMax = 20000.f) const {
        const float3 oc = ray_origin - center;
        const float half_b = dot(ray_direction, oc);
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

        const float3 p = ray_origin + t * ray_direction;
        tMin = t;
        return true;
    }
};