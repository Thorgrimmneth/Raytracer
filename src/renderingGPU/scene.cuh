#pragma once

#include "objects/sphere.cuh"
#include "objects/plane.cuh"
#include "objects/triangle_mesh.cuh"
#include "materials/material.cuh"
#include "objectsUtils/aabb.cuh"
#include "objectsUtils/bvh_scene.cuh"
#include "objects/implicitSphere.cuh"
#include "utils/quaternion.cuh"

struct Light;

struct CudaScene
{
    BVHScene bvhScene;
    Sphere *spheres;
    Plane *planes;
    TriangleMesh *triangleMeshes;
    ImplicitSphere *implicitSpheres;
    Material *materials;
    BaseObject *primitives;
    Light *lights;

    int nbSpheres;
    int nbPlanes;
    int nbTriangleMeshes;
    int nbMaterials;
    int nbLights;
    int nbImplicitSpheres;

    __host__ void uploadObjects(
        std::vector<Sphere> spheresGPU,
        std::vector<Plane> planesGPU,
        std::vector<TriangleMesh> triangleMeshesGPU,
        std::vector<BaseObject> primitivesGPU,
        std::vector<ImplicitSphere> implicitSpheresGPU);

    __host__ void uploadLights(
        std::vector<Light> lightsGPU);

    __host__ void uploadMaterials(std::vector<Material> materialsGPU);

    __device__ bool intersect(const Ray &, float, float, HitRecord &) const;

    __device__ bool intersectAny(const Ray &, float, float) const;

    __device__
    float lightPdf(
    const float3& origin,
    const float3& dir) const;

};

void sceneSize();

CudaScene spheresScene(float4 sunDir);

CudaScene implicitSpheresScene(float4 sunDir);

CudaScene singleObject(float4 sunDir);

struct MeshAndPrimitive{
    TriangleMesh mesh;
    BaseObject prim;

    MeshAndPrimitive(TriangleMesh p_mesh, float3 min, float3 max, ObjectType type, int index) : prim(BaseObject(min, max, type, index)), mesh(p_mesh) {}
};

__host__
MeshAndPrimitive loadTriangleMesh(const std::string& p_path, int materialIndex, int index, float3 scale, Quaternion rotation, float3 translation);