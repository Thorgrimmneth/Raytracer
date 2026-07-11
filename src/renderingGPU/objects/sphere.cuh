#pragma once

#include "../../utils/rngCPU.hpp"
#include "../utils/op.cuh"
#include <optix.h>
#include <optix_stubs.h>

struct Sphere
{
    float radius;
    int materialIndex;

    static Sphere createRandomSphere(int materialIndex)
    {
        float radius = randomFloat() * 2.f + 0.1f;
        return create(radius, materialIndex);
    }

    static Sphere create(float r, int m) { return {r, m}; }

    inline OptixAabb computeWorldAABB(const float3 &translation) const
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

    D_FORCEINLINE float3 getNormal(const float3 &point) const
    {
        float l = length(point);
        float4 normal = make_float4(l - radius, point / l);
        return make_float3(normal.y, normal.z, normal.w);
    }

    __device__ float sdf(const float3 &point) const { return length(point) - radius; }
};