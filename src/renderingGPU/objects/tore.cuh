#pragma once
#include "../../utils/rngCPU.hpp"
#include <optix.h>
#include <optix_stubs.h>
#include "../utils/op.cuh"

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

    OptixAabb computeAABB(const Matrix3x3 &rotation, const float3 &translation) const
    {
        // Conservative bounding box by transforming all 8 corners of the local AABB
        float3 extents = make_float3(radiusExter + radiusInter, radiusInter, radiusExter + radiusInter);
        
        // Define the 8 corners of the local AABB
        float3 corners[8] = {
            center1 + make_float3(-extents.x, -extents.y, -extents.z),
            center1 + make_float3(-extents.x, -extents.y, +extents.z),
            center1 + make_float3(-extents.x, +extents.y, -extents.z),
            center1 + make_float3(-extents.x, +extents.y, +extents.z),
            center1 + make_float3(+extents.x, -extents.y, -extents.z),
            center1 + make_float3(+extents.x, -extents.y, +extents.z),
            center1 + make_float3(+extents.x, +extents.y, -extents.z),
            center1 + make_float3(+extents.x, +extents.y, +extents.z)
        };
        Matrix3x3 localToWorld = rotation.transpose();
        // Transform all corners and compute the bounding box
        float3 minPoint = transform(localToWorld, corners[0]) + translation;
        float3 maxPoint = minPoint;
        
        for (int i = 1; i < 8; ++i) {
            float3 transformed = transform(localToWorld, corners[i]) + translation;
            minPoint = make_float3(
                fminf(minPoint.x, transformed.x),
                fminf(minPoint.y, transformed.y),
                fminf(minPoint.z, transformed.z)
            );
            maxPoint = make_float3(
                fmaxf(maxPoint.x, transformed.x),
                fmaxf(maxPoint.y, transformed.y),
                fmaxf(maxPoint.z, transformed.z)
            );
        }
        
        OptixAabb aabb;
        aabb.minX = minPoint.x;
        aabb.minY = minPoint.y;
        aabb.minZ = minPoint.z;
        aabb.maxX = maxPoint.x;
        aabb.maxY = maxPoint.y;
        aabb.maxZ = maxPoint.z;
        return aabb;
    }

    __device__ float sdf(const float3 &point) const
    {
        float3 p = point - center1;
        float3 q = make_float3(length(make_float3(p.x, 0.f, p.z)) - radiusExter, p.y, 0.f);
        return length(q) - radiusInter;
    }
};