#pragma once

#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../objects/triangle_mesh_geometry.cuh"
#include "../raytracingUtils/ray.cuh"

struct BVH{
    AABB bbox;
    int left = -1;
    int right = -1;
    int firstTriangleIndex;
    int lastTriangleIndex;

    __device__
    inline bool isLeaf() const { return ( left == -1); }
};

// Forward declarations for BVH functions
__host__
BVH* buildBVH(
    TriangleMeshGeometry* triangles,
    int triangleCount,
    float3* vertices,
    float3* normals,
    float2* uvs,
    int& outNodeCount);
