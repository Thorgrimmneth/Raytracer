#pragma once

#include "objects/sphere.cuh"
#include "objects/plane.cuh"
#include "objects/triangle_mesh.cuh"
#include "materials/material.cuh"
#include "objectsUtils/aabb.cuh"
#include "objectsUtils/bvh_scene.cuh"
#include "objects/implicitSphere.cuh"
#include "utils/quaternion.cuh"
#include "objectsUtils/bvh_scene.cuh"
#include "raytracingUtils/ray.cuh"
#include "raytracingUtils/hitrecord.cuh"
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

    __device__ __forceinline__ 
    bool intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const
    {
        float tMax = p_tMax;
        bool hit = false;
        for(int i = 0; i < nbPlanes; ++i)
        {
            HitRecord planeHit;

            if (planes[i].intersect(p_ray, p_tMin, tMax, planeHit))
            {
                tMax = planeHit.distance;
                p_hitRecord = planeHit;

                p_hitRecord.objectType = HIT_PLANE;
                p_hitRecord.objectIndex = i;

                hit = true;
            }
        }
        if (bvhScene.intersect(p_ray, p_tMin, tMax, p_hitRecord))
        {
            tMax = p_hitRecord.distance; // update tMax to conserve the nearest hit
            hit = true;
        }
        
        return hit;
    }

    __device__ __forceinline__ 
    bool intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
    {
        for(int i = 0; i < nbPlanes; ++i)
        {
            if (planes[i].intersectAny(p_ray, p_tMin, p_tMax, materials))
            {
                return true;
            }
        }
        if(bvhScene.intersectAny(p_ray, p_tMin, p_tMax, materials))
        {
            return true;
        }
        return false;
    }

    __device__ __forceinline__
    float lightPdf(
    const float3& origin,
    const float3& dir) const
    {
        Ray ray(origin, dir);

        HitRecord hit;

        if (!intersect(ray, 1e-4f, 1e30f, hit))
            return 0.0f;

        const MaterialType matType = materials[hit.materialIndex].type();

        if (matType != MaterialType::EMISSIVE)
            return 0.0f;

        const float dist2 = hit.distance * hit.distance;
        float pdf = 0.0f;

        if (hit.objectType == HIT_SPHERE)
        {
            const Sphere& s = spheres[hit.objectIndex];

            const float3 toSurface = hit.point - s.center1;
            const float invRadius = 1.0f / s.radius;

            const float cosTheta = fmaxf(dot(toSurface, -dir) * invRadius, 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            const float area = 4.0f * GPUPIf * s.radius * s.radius;
            pdf = dist2 / (area * cosTheta);
        }
        else if (hit.objectType == HIT_SPHERE_IMPLICIT)
        {
            const ImplicitSphere& s = implicitSpheres[hit.objectIndex];

            const float3 toSurface = hit.point - s.center1;
            const float invRadius = 1.0f / s.radius;

            const float cosTheta = fmaxf(dot(toSurface, -dir) * invRadius, 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            const float area = 4.0f * GPUPIf * s.radius * s.radius;
            pdf = dist2 / (area * cosTheta);
        }
        else if (hit.objectType == HIT_TRIANGLE_MESH)
        {
            const TriangleMesh& mesh = triangleMeshes[hit.objectIndex];

            const float cosTheta = fmaxf(dot(hit.normal, -dir), 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            pdf = dist2 / (mesh.meshArea * cosTheta);
        }
        else
        {
            return 0.0f;
        }

        return pdf * (1.0f / nbLights);
    };

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