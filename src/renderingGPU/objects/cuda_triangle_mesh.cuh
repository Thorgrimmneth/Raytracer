#pragma once

#include "../objectsUtils/cuda_bvh.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"
#include "cuda_triangle_mesh_geometry.cuh"
#include "../materials/cuda_material.cuh"
//pas besoin de triangle_mesh_geometry parce qu'on parcourt le bvh pour l'intersection
struct TriangleMesh{
    BVH* bvhNodes;
    int bvhNodeCount;

    TriangleMeshGeometry* triangles;
    int triangleCount;
    float* triangleAreaCdf;
    float meshArea;

    float3* vertices;
    float3* normals;
    float2* uvs;

    int vertexCount;
    int materialIndex;

    __device__ __noinline__
    bool intersect( const Ray & p_ray,
					const float p_tMin,
					const float p_tMax,
					HitRecord & p_hitRecord) const;

    __device__ __noinline__
    bool intersectAny( const Ray & p_ray,
					   const float p_tMin,
					   const float p_tMax,
                    const Material* materials) const;
};