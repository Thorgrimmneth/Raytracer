#pragma once

namespace RT {
    class Scene;
}

#include "objects/cuda_sphere.cuh"
#include "objects/cuda_plane.cuh"
#include "objects/cuda_triangle_mesh.cuh"
#include "materials/cuda_material.cuh"
#include "lights/cuda_light.cuh"
#include "objectsUtils/cuda_aabb.cuh"
#include "objectsUtils/cuda_bvh_scene.cuh"

struct CudaScene
{
    BVHScene bvhScene;
    Sphere* spheres;
    Plane* planes;
    TriangleMesh* triangleMeshes;
    Material* materials;
    BaseObject* primitives;
    Light* lights;
    float3* vertices;

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

CudaScene uploadSceneToGPU(const RT::Scene& scene, float4 sunDir);
