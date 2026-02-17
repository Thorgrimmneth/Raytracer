#pragma once

#include "cuda_bvh.cuh"
#include "cuda_hitrecord.cuh"
#include "cuda_triangle_mesh_geometry.cuh"

//pas besoin de triangle_mesh_geometry parce qu'on parcourt le bvh pour l'intersection
struct TriangleMesh{
    BVH* bvhNodes;
    int bvhNodeCount;

    TriangleMeshGeometry* triangles;
    int triangleCount;

    float4* vertices;
    float4* normals;
    float2* uvs;

    int vertexCount;
    int materialIndex;

    __device__
    bool intersect( const Ray & p_ray,
					const float p_tMin,
					const float p_tMax,
					HitRecord & p_hitRecord,
                    float4* vertices) const;

    __device__
    bool intersectAny( const Ray & p_ray,
					   const float p_tMin,
					   const float p_tMax,
                       float4* vertices) const;
};