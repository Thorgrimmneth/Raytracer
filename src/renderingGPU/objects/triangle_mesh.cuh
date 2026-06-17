#pragma once

struct TriangleMesh
{
    uint3 *triangles = nullptr;
    float3 *vertices = nullptr;
    float3 *normals = nullptr;
    float2 *uvs = nullptr;

    int triangleCount = 0;
    int vertexCount = 0;

    int materialIndex = 0;

    float meshArea = 0.0f;
    float *triangleAreaCdf = nullptr;

};