#pragma once

#include "../utils/macro.cuh"
#include "mesh_loader.cuh"

#include "../lights/light.cuh"
#include "../materials/material.cuh"

#include "../objects/implicit_sphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/sphere.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../objectsUtils/aabb.cuh"
#include "../objectsUtils/bvh_scene.cuh"

#include "../raytracingUtils/ray.cuh"


#include "../../../devicePrograms/optix_launch_params_manager.h"
#include "../optix/optix_context.h"
#include "../optix/optix_gas.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"
#include "../optix/optix_sbt_manager.h"
#include "../../../devicePrograms/launch_params.cuh"
#include "scene_helper.cuh"

struct Light;

struct OptixSceneData
{
    OptixPipeline pipeline = nullptr;

    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_launchParams = 0;
    LaunchParams launchParams = {};

    OptixTraversableHandle gasHandle = 0;
    CUdeviceptr d_gasBuffer = 0;

    OptixSceneData() = default;
    OptixSceneData(OptixPipeline p, OptixShaderBindingTable s, CUdeviceptr lp, LaunchParams lpStruct,
                   OptixTraversableHandle gasH, CUdeviceptr gasBuf)
        : pipeline(p), sbt(s), d_launchParams(lp), launchParams(lpStruct), gasHandle(gasH), d_gasBuffer(gasBuf)
    {
    }

    void destroy()
    {
        if (pipeline)
            optixPipelineDestroy(pipeline);
        if (d_launchParams)
            CUDA_CHECK(cudaFree((void *)d_launchParams));
        if (d_gasBuffer)
            CUDA_CHECK(cudaFree((void *)d_gasBuffer));
    }
};

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
    float *lightProbabilities;
    float *lightCumulativeWeights;

    OptixSceneData optixData;

    int nbSpheres;
    int nbPlanes;
    int nbTriangleMeshes;
    int nbMaterials;
    int nbLights;
    int nbImplicitSpheres;
    
    void sceneSize(CudaSceneHelper &helper);

    HOST void uploadObjects(CudaSceneHelper &helper);

    HOST void uploadLights(CudaSceneHelper &helper);

    HOST void uploadMaterials(CudaSceneHelper &helper);

    D_FORCEINLINE bool intersect(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax, OptixHit &p_hitRecord) const
    {
        float tMax = p_tMax;
        bool hit = false;
        for (int i = 0; i < nbPlanes; ++i)
        {
            OptixHit planeHit;

            if (planes[i].intersect(origin, direction, p_tMin, tMax, planeHit))
            {
                tMax = planeHit.t;
                p_hitRecord = planeHit;

                p_hitRecord.objectType = HIT_PLANE;
                p_hitRecord.objectIndex = i;

                hit = true;
            }
        }
        if (bvhScene.intersect(origin, direction, p_tMin, tMax, p_hitRecord))
        {
            tMax = p_hitRecord.t; // update tMax to conserve the nearest hit
            hit = true;
        }

        return hit;
    }

    D_FORCEINLINE bool intersectAny(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax) const
    {
        for (int i = 0; i < nbPlanes; ++i)
        {
            if (planes[i].intersectAny(origin, direction, p_tMin, p_tMax, materials))
            {
                return true;
            }
        }
        if (bvhScene.intersectAny(origin, direction, p_tMin, p_tMax, materials))
        {
            return true;
        }
        return false;
    }

    D_FORCEINLINE float3 traceShadowRay(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax) const
    {
        float3 shadowColor = make_float3(1.f);
        float remainingDistance = p_tMax;
        float3 originT = origin;
        float3 directionT = direction;

        // Trace through up to 2 transparent surfaces
        for (int bounce = 0; bounce < 2; ++bounce)
        {
            OptixHit hit;

            if (!intersect(originT, directionT, p_tMin + 1e-4f, remainingDistance - 1e-4f, hit))
            {
                // No hit = ray reached the light
                return shadowColor;
            }

            const Material &mtl = materials[hit.materialIndex];
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
                originT = hit.position + directionT * 1e-4f;
                remainingDistance -= hit.t;
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

    D_FORCEINLINE float lightPdf(const float3 &origin, const float3 &dir) const
    {
        OptixHit hit;

        if (!intersect(origin, dir, 1e-4f, 1e30f, hit))
            return 0.0f;

        const MaterialType matType = materials[hit.materialIndex].type();

        if (matType != MaterialType::EMISSIVE)
            return 0.0f;

        const float dist2 = hit.t * hit.t;
        float pdf = 0.0f;

        if (hit.objectType == HIT_SPHERE)
        {
            const Sphere &s = spheres[hit.objectIndex];

            const float3 toSurface = hit.position - s.getCenter1();
            const float invRadius = 1.0f / s.getRadius();

            const float cosTheta = fmaxf(dot(toSurface, -dir) * invRadius, 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            const float area = 4.0f * GPUPIf * s.getRadius() * s.getRadius();
            pdf = dist2 / (area * cosTheta);
        }
        else if (hit.objectType == HIT_SPHERE_IMPLICIT)
        {
            const ImplicitSphere &s = implicitSpheres[hit.objectIndex];

            const float3 toSurface = hit.position - s.getCenter1();
            const float invRadius = 1.0f / s.getRadius();

            const float cosTheta = fmaxf(dot(toSurface, -dir) * invRadius, 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            const float area = 4.0f * GPUPIf * s.getRadius() * s.getRadius();
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