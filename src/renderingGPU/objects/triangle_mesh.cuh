#pragma once

#include "../utils/op.cuh"

// Shared geometry data - loaded once per mesh file
struct MeshGeometry
{
    uint3 *triangles = nullptr;
    float3 *vertices = nullptr;
    float3 *normals = nullptr;
    float2 *uvs = nullptr;

    int triangleCount = 0;
    int vertexCount = 0;

    float meshArea = 0.0f;
    float *triangleAreaCdf = nullptr;
};

// Per-instance data - one per mesh instance in the scene
struct MeshInstance
{
    int geometryIndex = 0;     // Index into the geometries array
    int materialIndex = 0;      // Material for this instance
    float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // 3x4 transformation matrix (computed from rotation/scale/translation)
    float worldArea = 1.f;
    int lightIndex = -1; // Index into the lights array, if this instance is emissive
};