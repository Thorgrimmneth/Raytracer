#pragma once

namespace RT {
    class Scene;
}

#include "objectGPU/cuda_sphere.cuh"
#include "objectGPU/cuda_plane.cuh"
#include "objectGPU/cuda_triangle_mesh.cuh"
#include "cuda_material.cuh"
#include "cuda_light.cuh"
#include "objectGPU/cuda_aabb.cuh"

struct BVHNodeGPU
{
	AABB bbox;
	int left;
	int right;
	int firstTriangle;
	int lastTriangle;
};

struct CudaScene
{
    Sphere* spheres;
    Plane* planes;
    TriangleMesh* triangleMeshes;
    Material* materials;
    Light* lights;
    float4* vertices;

    int nbSpheres;
    int nbPlanes;
    int nbTriangleMeshes;
    int nbMaterials;
    int nbLights;

    __device__
    bool intersect(const Ray&, float, float, HitRecord&) const;

    __device__
    bool intersectAny(const Ray&, float, float) const;
};

CudaScene uploadSceneToGPU(const RT::Scene& scene);
