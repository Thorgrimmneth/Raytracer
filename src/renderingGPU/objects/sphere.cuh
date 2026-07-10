#pragma once

#include "../../utils/rngCPU.hpp"
#include "../utils/op.cuh"
#include <optix.h>
#include <optix_stubs.h>

struct Sphere
{
    float3 center1;
    float radius;
    int materialIndex;

    static Sphere createRandomSphere(int materialIndex)
    {
        float3 center = make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        float radius = randomFloat() * 2.f + 0.1f;
        return create(center, radius, materialIndex);
    }

    static Sphere create(float3 c, float r, int m) { return {c, r, m}; }

    OptixAabb computeAABB() const
    {

        OptixAabb aabb;
        aabb.minX = center1.x - radius;
        aabb.minY = center1.y - radius;
        aabb.minZ = center1.z - radius;
        aabb.maxX = center1.x + radius;
        aabb.maxY = center1.y + radius;
        aabb.maxZ = center1.z + radius;
        return aabb;
    }

    __device__ float sdf(const float3 &point) const { return length(point - center1) - radius; }
};