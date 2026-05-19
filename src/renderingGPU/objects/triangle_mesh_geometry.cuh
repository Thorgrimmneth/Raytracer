#pragma once

#include "../objectsUtils/aabb.cuh"

class TriangleMesh;

struct TriangleMeshGeometry{
    int i0, i1, i2;

    __device__
    bool intersect(
        const Ray& ray,
        float tMin,
        float tMax,
        float& t,
        float2& uv,
        const float3* __restrict__ vertices) const;

    __device__
    const float3 computeSmoothNormal( const float2 & p_uv, const float3* normals ) const;
};