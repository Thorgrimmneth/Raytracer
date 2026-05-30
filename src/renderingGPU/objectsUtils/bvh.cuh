#pragma once

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../utils/macro.cuh"

#include "aabb.cuh"

#include "../objects/triangle_mesh_geometry.cuh"

#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"

struct BVH {
    AABB bbox;
    int left = -1;
    int right = -1;

    // Plage dans un tableau d'indices de triangles
    int firstRefIndex = -1;
    int refCount = 0;

    D_FORCEINLINE
    bool isLeaf() const { return left == -1; }
};

HOST
BVH* buildBVH(
    TriangleMeshGeometry* triangles,
    int triangleCount,
    float4* vertices,
    float4* normals,
    float2* uvs,
    int& outNodeCount,
    int*& outTriangleRefIndices,
    int& outRefCount);
