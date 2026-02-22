#pragma once

#include "../objectsUtils/cuda_aabb.cuh"

class TriangleMesh;

struct TriangleMeshGeometry{
    int i0, i1, i2;

    __device__
    bool intersect( const Ray & p_ray, float & p_t, float2 & p_uv,float3* vertices ) const;

    __device__
    const float3 computeSmoothNormal( const float2 & p_uv, const float3* normals ) const;
};