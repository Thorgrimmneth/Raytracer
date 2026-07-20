#pragma once
#include "../../utils/rng_cpu.hpp"
#include "../utils/op.cuh"
#include <optix.h>
#include <optix_stubs.h>

struct Tore
{
    float radiusInter;
    float radiusExter;
    int materialIndex;

    static Tore createRandomTore(int materialIndex)
    {
        float radiusInter = randomFloat() * 0.5f + 0.1f;
        float radiusExter = radiusInter + randomFloat() * 1.0f + 0.1f;
        return create(radiusInter, radiusExter, materialIndex);
    }

    static Tore create(float rInter, float rExter, int m) { return {rInter, rExter, m}; }

    inline OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation) const
    {
        // Conservative bounding box by transforming all 8 corners of the local AABB
        float3 extents = make_float3(radiusExter + radiusInter, radiusInter, radiusExter + radiusInter);
        // Define the 8 corners of the local AABB
        float3 corners[8] = {
            make_float3(-extents.x, -extents.y, -extents.z), make_float3(-extents.x, -extents.y, +extents.z),
            make_float3(-extents.x, +extents.y, -extents.z), make_float3(-extents.x, +extents.y, +extents.z),
            make_float3(+extents.x, -extents.y, -extents.z), make_float3(+extents.x, -extents.y, +extents.z),
            make_float3(+extents.x, +extents.y, -extents.z), make_float3(+extents.x, +extents.y, +extents.z)};
        Matrix3x3 localToWorld = rotation.transpose();
        // Transform all corners and compute the bounding box
        float3 minPoint = transform(localToWorld, corners[0]);
        float3 maxPoint = minPoint;

        for (int i = 1; i < 8; ++i)
        {
            float3 transformed = transform(localToWorld, corners[i]) + translation;
            minPoint = make_float3(fminf(minPoint.x, transformed.x), fminf(minPoint.y, transformed.y),
                                   fminf(minPoint.z, transformed.z));
            maxPoint = make_float3(fmaxf(maxPoint.x, transformed.x), fmaxf(maxPoint.y, transformed.y),
                                   fmaxf(maxPoint.z, transformed.z));
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

    HD_FORCEINLINE OptixAabb computeAABB(const Matrix3x3 &rotation) const
    {
        float e = radiusExter + radiusInter;

        return {-e, -radiusInter, -e, e, radiusInter, e};
    }

    D_FORCEINLINE float3 getNormal(const float3 &point, const Matrix3x3 &rotation) const
    {
        // Transform the point to local space
        float3 localPoint = transform(rotation, point);
        float3 q = make_float3(length(make_float3(localPoint.x, 0.f, localPoint.z)) - radiusExter, localPoint.y, 0.f);
        float l = length(q);
        if (l == 0.f)
            return make_float3(0.f, 1.f, 0.f); // Arbitrary normal if on the surface
        float4 normal = make_float4(l - radiusInter, q / l);
        // Transform the normal back to world space
        float3 worldNormal = transform(rotation.transpose(), make_float3(normal.y, normal.z, normal.w));
        return normalize(worldNormal);
    }

    __device__ float sdf(const float3 &point) const
    {
        float3 p = point;
        float3 q = make_float3(length(make_float3(p.x, 0.f, p.z)) - radiusExter, p.y, 0.f);
        return length(q) - radiusInter;
    }
};