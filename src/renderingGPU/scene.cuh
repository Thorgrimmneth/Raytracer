#pragma once

#include "objects/sphere.cuh"
#include "objects/plane.cuh"
#include "objects/triangle_mesh.cuh"
#include "materials/material.cuh"
#include "objectsUtils/aabb.cuh"
#include "objectsUtils/bvh_scene.cuh"

struct Light;

struct CudaScene
{
    BVHScene bvhScene;
    Sphere *spheres;
    Plane *planes;
    TriangleMesh *triangleMeshes;
    Material *materials;
    BaseObject *primitives;
    Light *lights;
    float3 *vertices;

    int nbSpheres;
    int nbPlanes;
    int nbTriangleMeshes;
    int nbMaterials;
    int nbLights;

    __host__ void uploadObjects(
        std::vector<Sphere> spheresGPU,
        std::vector<Plane> planesGPU,
        std::vector<TriangleMesh> triangleMeshesGPU,
        std::vector<float3> verticesGPU,
        std::vector<BaseObject> primitivesGPU);

    __host__ void uploadLights(
        std::vector<Light> lightsGPU);

    __host__ void uploadMaterials(std::vector<Material> materialsGPU);

    __device__ bool intersect(const Ray &, float, float, HitRecord &) const;

    __device__ bool intersectAny(const Ray &, float, float) const;

    

};

CudaScene spheresScene(float4 sunDir);
