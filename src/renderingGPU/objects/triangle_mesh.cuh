#pragma once

#include "../objectsUtils/bvh.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "triangle_mesh_geometry.cuh"
#include "../materials/material.cuh"

struct TriangleMesh {
    TriangleMeshGeometry* triangles = nullptr;
    float3* vertices = nullptr;
    float3* normals = nullptr;
    float2* uvs = nullptr;

    int triangleCount = 0;
    int vertexCount = 0;

    BVH* bvhNodes = nullptr;
    int bvhNodeCount = 0;

    int* triangleRefIndices = nullptr;
    int refCount = 0;

    float meshArea = 0.0f;
    float* triangleAreaCdf = nullptr;
    int materialIndex = 0;

    __device__ bool intersect(
        const Ray& p_ray,
        float p_tMin,
        float p_tMax,
        HitRecord& p_hitRecord) const;

    __device__ bool intersectAny(
        const Ray& p_ray,
        float p_tMin,
        float p_tMax,
        const Material* materials) const;
};