#pragma once
#include "../../utils/rngCPU.hpp"
#include <optix.h>
#include <optix_stubs.h>

struct Tore
{
    float3 center1;
    float radiusInter;
    float radiusExter;
    int materialIndex;

    static Tore createRandomTore(int materialIndex)
    {
        float3 center = make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        float radiusInter = randomFloat() * 0.5f + 0.1f;
        float radiusExter = radiusInter + randomFloat() * 1.0f + 0.1f;
        return create(center, radiusInter, radiusExter, materialIndex);
    }

    static Tore create(float3 c, float rInter, float rExter, int m)
    {
        return {c, rInter, rExter, m};
    }

    OptixAabb computeAABB() const
    {
        OptixAabb aabb;
        aabb.minX = center1.x - (radiusExter + radiusInter);
        aabb.minY = center1.y - radiusExter;
        aabb.minZ = center1.z - (radiusExter + radiusInter);
        aabb.maxX = center1.x + (radiusExter + radiusInter);
        aabb.maxY = center1.y + radiusExter;
        aabb.maxZ = center1.z + (radiusExter + radiusInter);
        return aabb;
    }

    __device__ float sdf(const float3 &point) const
    {
        float3 p = point - center1;
        float3 q = make_float3(length(make_float3(p.x, 0.f, p.z)) - radiusExter, p.y, 0.f);
        return length(q) - radiusInter;
    }
};