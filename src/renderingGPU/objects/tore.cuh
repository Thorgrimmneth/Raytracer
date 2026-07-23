#pragma once
#include "../../utils/rng_cpu.hpp"
#include "../utils/rng.cuh"
#include "../utils/defines.cuh"
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

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation) const
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

    HD_FORCEINLINE OptixAabb computeAABB() const
    {
        float e = radiusExter + radiusInter;

        return {-e, -radiusInter, -e, e, radiusInter, e};
    }

    D_FORCEINLINE float3 getNormal(const float3 &point) const
    {
        float h = length(make_float3(point.x, 0.f, point.z));
        float4 q = make_float4(length(make_float2(h - radiusExter, point.y)) - radiusInter, normalize(point * make_float3(h - radiusExter, h, h - radiusExter)));
        return normalize(make_float3(q.y, q.z, q.w));
    }

    __device__ float sdf(const float3 &point) const
    {
        float3 p = point;
        float3 q = make_float3(length(make_float3(p.x, 0.f, p.z)) - radiusExter, p.y, 0.f);
        return length(q) - radiusInter;
    }

    D_FORCEINLINE float getMajorRadius() const { return radiusExter; }
    D_FORCEINLINE float getMinorRadius() const { return radiusInter; }

    // Sample a random point on the torus surface
    D_FORCEINLINE void sampleSurfacePoint(float3 &p_point, float3 &p_normal, RNG &rng) const
    {
        // Sample position on the torus: major angle and minor angle
        float majorAngle = 2.f * GPUPIf * rng.nextFloat();
        float minorAngle = 2.f * GPUPIf * rng.nextFloat();
            
        // Parametric torus surface
        float cosMajor = cosf(majorAngle);
        float sinMajor = sinf(majorAngle);
        float cosMinor = cosf(minorAngle);
        float sinMinor = sinf(minorAngle);
        
        // Distance from major circle to current point
        float distFromMajor = radiusExter + radiusInter * cosMinor;
        
        // Surface position
        p_point = make_float3(
            distFromMajor * cosMajor,
            radiusInter * sinMinor,
            distFromMajor * sinMajor
        );
        
        // Surface normal (pointing outward)
        p_normal = normalize(make_float3(
            cosMinor * cosMajor,
            sinMinor,
            cosMinor * sinMajor
        ));
    }
};