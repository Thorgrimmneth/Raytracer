#pragma once

#include "../utils/macro.cuh"
#include "mesh_loader.cuh"

#include "../materials/material.cuh"
#include "../lights/light.cuh"

#include "../objects/implicitSphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/sphere.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../objectsUtils/aabb.cuh"
#include "../objectsUtils/bvh_scene.cuh"

#include "../raytracingUtils/hitrecord.cuh"
#include "../raytracingUtils/ray.cuh"



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

    void sceneSize(std::vector<BaseObject> primitivesGPU);
    
    HOST 
    void uploadObjects(std::vector<Sphere> spheresGPU, 
                            std::vector<Plane> planesGPU,
                            std::vector<TriangleMesh> triangleMeshesGPU, 
                            std::vector<BaseObject> primitivesGPU,
                            std::vector<ImplicitSphere> implicitSpheresGPU);

    HOST 
    void uploadLights(std::vector<Light> lightsGPU);

    HOST 
    void uploadMaterials(std::vector<Material> materialsGPU);

    D_FORCEINLINE 
    bool intersect(const Ray &p_ray, const float p_tMin, const float p_tMax,
                                              HitRecord &p_hitRecord) const
    {
        float tMax = p_tMax;
        bool hit = false;
        for (int i = 0; i < nbPlanes; ++i)
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

    D_FORCEINLINE
    bool intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
    {
        for (int i = 0; i < nbPlanes; ++i)
        {
            if (planes[i].intersectAny(p_ray, p_tMin, p_tMax, materials))
            {
                return true;
            }
        }
        if (bvhScene.intersectAny(p_ray, p_tMin, p_tMax, materials))
        {
            return true;
        }
        return false;
    }

    D_FORCEINLINE
    float3 traceShadowRay(const Ray &p_ray, const float p_tMin, const float p_tMax) const
    {
        float3 shadowColor = make_float3(1.f);
        Ray currentRay = p_ray;
        float remainingDistance = p_tMax;
        
        // Trace through up to 2 transparent surfaces
        for (int bounce = 0; bounce < 2; ++bounce)
        {
            HitRecord hit;
            
            if (!intersect(currentRay, p_tMin + 1e-4f, remainingDistance - 1e-4f, hit))
            {
                // No hit = ray reached the light
                return shadowColor;
            }
            
            const Material& mtl = materials[hit.materialIndex];
            MaterialType matType = mtl.type();
            
            // Check if material is transparent
            if (matType == TRANSPARENT)
            {
                // Tint shadow ray with material transmission color
                float3 transmission = mtl.computeTransmission();
                shadowColor *= transmission;
                
                // Early termination: if transmission becomes negligible, stop bouncing
                float transAlpha = fmaxf(shadowColor.x, fmaxf(shadowColor.y, shadowColor.z));
                if (transAlpha < 0.001f)
                {
                    return make_float3(0.f);
                }
                
                // Continue ray from hit point toward light
                currentRay = Ray(hit.point + currentRay.direction * 1e-4f, currentRay.direction, currentRay.time);
                remainingDistance -= hit.distance;
            }
            else if (matType == EMISSIVE)
            {
                return shadowColor;
            }
            else
            {
                // Opaque material blocks shadow completely
                return make_float3(0.f);
            }
        }
        
        // After max bounces, assume ray reached the light
        return shadowColor;
    }

    D_FORCEINLINE
    float lightPdf(const float3 &origin, const float3 &dir) const
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
            const Sphere &s = spheres[hit.objectIndex];

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
            const ImplicitSphere &s = implicitSpheres[hit.objectIndex];

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
            const TriangleMesh &mesh = triangleMeshes[hit.objectIndex];

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

CudaScene spheresScene(float4 sunDir);

CudaScene implicitSpheresScene(float4 sunDir);

CudaScene singleObject(float4 sunDir);