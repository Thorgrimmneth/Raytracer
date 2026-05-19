#pragma once

#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../objects/triangle_mesh_geometry.cuh"
#include "../raytracingUtils/ray.cuh"

struct BVH {
    AABB bbox;
    int left = -1;
    int right = -1;

    // Plage dans un tableau d'indices de triangles
    int firstRefIndex = -1;
    int refCount = 0;

    __host__ __device__
    inline bool isLeaf() const { return left == -1; }
};

// Forward declarations for BVH functions
__host__
BVH* buildBVH(
    TriangleMeshGeometry* triangles,
    int triangleCount,
    float3* vertices,
    float3* normals,
    float2* uvs,
    int& outNodeCount,
    int*& outTriangleRefIndices,
    int& outRefCount);
