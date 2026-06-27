#pragma once

#include "../utils/simplifiedDef.cuh"
#include "mesh_loader.cuh"

#include <vector>

#include "../lights/light.cuh"
#include "../materials/material.cuh"

#include "../objects/implicit_sphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/sphere.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../objectsUtils/aabb.cuh"
#include "../objectsUtils/bvh_scene.cuh"

#include "../raytracingUtils/ray.cuh"

#include "../../../devicePrograms/launch_radiance_params.cuh"
#include "../../../devicePrograms/launch_shadow_params.cuh"
#include "../../../devicePrograms/optix_launch_params_manager.h"
#include "../optix/optix_context.h"
#include "../optix/optix_gas.h"
#include "../optix/optix_ias.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"
#include "../optix/optix_sbt_manager.h"
#include "../utils/computeTransform.cuh"
#include "init_optix.cuh"
#include "scene_helper.cuh"
#include "sun_helper.h"

struct Light;

template <typename LaunchParamsT> struct OptixPassData
{
    OptixPipelineManager pipeline;
    OptixProgramGroupManager programGroups;
    OptixSBTManager sbt;
    OptixLaunchParamsManager<LaunchParamsT> launchParams;
};

struct CudaScene
{
    Sphere *spheres;
    Plane *planes;
    MeshGeometry *meshGeometries; // Shared geometry data (loaded once per file)
    MeshInstance *meshInstances;  // Per-instance data (transform, material)
    ImplicitSphere *implicitSpheres;
    Material *materials;
    BaseObject *primitives;
    Light *lights;
    float *lightProbabilities;
    float *lightCumulativeWeights;

    OptixTraversableHandle iasHandle = 0;
    CUdeviceptr d_iasBuffer = 0;
    std::vector<OptixGAS> gasList;

    OptixPassData<LaunchRadianceParams> radiancePass;
    OptixPassData<LaunchShadowParams> shadowPass;

    int nbSpheres;
    int nbPlanes;
    int nbMeshes;
    int nbMeshGeometries; // Number of unique mesh geometries
    int nbMeshInstances;  // Number of mesh instances
    int nbMaterials;
    int nbLights;
    int nbImplicitSpheres;

    inline HOST void destroy()
    {
        // Libération des géométries
        if (meshGeometries)
        {
            for (int i = 0; i < nbMeshGeometries; ++i)
            {
                CUDA_CHECK(cudaFree(meshGeometries[i].vertices));
                CUDA_CHECK(cudaFree(meshGeometries[i].normals));
                CUDA_CHECK(cudaFree(meshGeometries[i].uvs));
                CUDA_CHECK(cudaFree(meshGeometries[i].triangles));
                CUDA_CHECK(cudaFree(meshGeometries[i].triangleAreaCdf));
            }

            CUDA_CHECK(cudaFree(meshGeometries));
        }

        CUDA_CHECK(cudaFree(meshInstances));

        CUDA_CHECK(cudaFree(spheres));
        CUDA_CHECK(cudaFree(planes));
        CUDA_CHECK(cudaFree(implicitSpheres));

        CUDA_CHECK(cudaFree(materials));

        CUDA_CHECK(cudaFree(primitives));

        CUDA_CHECK(cudaFree(lights));
        CUDA_CHECK(cudaFree(lightProbabilities));
        CUDA_CHECK(cudaFree(lightCumulativeWeights));

        // Remise à zéro
        meshGeometries = nullptr;
        meshInstances = nullptr;
        spheres = nullptr;
        planes = nullptr;
        implicitSpheres = nullptr;
        materials = nullptr;
        primitives = nullptr;
        lights = nullptr;
        lightProbabilities = nullptr;
        lightCumulativeWeights = nullptr;

        nbMeshGeometries = 0;
        nbMeshInstances = 0;
        nbMeshes = 0;
        nbSpheres = 0;
        nbPlanes = 0;
        nbImplicitSpheres = 0;
        nbMaterials = 0;
        nbLights = 0;
    }

    HOST void uploadObjects(CudaSceneHelper &helper);

    HOST void uploadLights(CudaSceneHelper &helper);

    HOST void uploadMaterials(CudaSceneHelper &helper);

    D_FORCEINLINE bool intersect(const float3 &origin, const float3 &direction, const float p_tMin, const float p_tMax,
                                 OptixHit &p_hitRecord) const
    {
        return false;
        // float tMax = p_tMax;
        // bool hit = false;
    }

    D_FORCEINLINE bool intersectAny(const float3 &origin, const float3 &direction, const float p_tMin,
                                    const float p_tMax) const
    {
        return false;
    }

    D_FORCEINLINE float3 traceShadowRay(const float3 &origin, const float3 &direction, const float p_tMin,
                                        const float p_tMax) const
    {

        float3 shadowColor = make_float3(1.f);
        return shadowColor;
        /*float remainingDistance = p_tMax;
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
        return shadowColor;*/
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
            const MeshInstance &inst = meshInstances[hit.objectIndex];
            const MeshGeometry &geom = meshGeometries[inst.geometryIndex];

            const float cosTheta = fmaxf(dot(hit.normal, -dir), 0.0f);

            if (cosTheta <= 0.0f)
                return 0.0f;

            pdf = dist2 / (geom.meshArea * cosTheta);
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

CudaScene singleObject(float4 sunDir, int rngmanip = 0);