#pragma once

#include "../utils/simplified_def.cuh"
#include "mesh_loader.cuh"

#include <vector>

#include "../lights/light.cuh"
#include "../materials/material.cuh"

#include "../objects/plane.cuh"
#include "../objects/sphere_sdf.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../raytracingUtils/ray.cuh"

#include "../optix/optix_context.h"
#include "../optix/optix_gas.h"
#include "../optix/optix_ias.h"
#include "../optix/optix_sbt_manager.h"
#include "../utils/compute_transform.cuh"
#include "../utils/optix_pass_data.cuh"
#include "init_optix.cuh"
#include "scene_helper.cuh"
#include "sun_helper.h"

struct Light;

struct Scene
{
    SphereSDF *spheres;
    Plane *planes;
    MeshGeometry *meshGeometries;
    MeshInstance *meshInstances;
    Material *materials;
    Light *lights;
    float *lightProbabilities;
    float *lightCumulativeWeights;
    SDFGeometry sdfGeometries;
    OptixTraversableHandle iasHandle = 0;
    CUdeviceptr d_iasBuffer = 0;
    std::vector<OptixGAS> gasList;

    int nbSpheres;
    int nbPlanes;
    int nbMeshes;
    int nbMeshGeometries; // Number of unique mesh geometries
    int nbMeshInstances;  // Number of mesh instances
    int nbMaterials;
    int nbLights;

    inline HOST void destroy()
    {
        std::cout << "Destroying Scene..." << std::endl;
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
            std::cout << "Destroyed " << nbMeshGeometries << " mesh geometries." << std::endl;
            CUDA_CHECK(cudaFree(meshGeometries));
        }
        std::cout << "Destroyed " << nbMeshGeometries << " mesh geometries." << std::endl;

        CUDA_CHECK(cudaFree(meshInstances));

        CUDA_CHECK(cudaFree(spheres));
        CUDA_CHECK(cudaFree(planes));

        CUDA_CHECK(cudaFree(materials));

        CUDA_CHECK(cudaFree(lights));
        CUDA_CHECK(cudaFree(lightProbabilities));
        CUDA_CHECK(cudaFree(lightCumulativeWeights));

        // Remise à zéro
        meshGeometries = nullptr;
        meshInstances = nullptr;
        spheres = nullptr;
        planes = nullptr;
        materials = nullptr;
        lights = nullptr;
        lightProbabilities = nullptr;
        lightCumulativeWeights = nullptr;

        nbMeshGeometries = 0;
        nbMeshInstances = 0;
        nbMeshes = 0;
        nbSpheres = 0;
        nbPlanes = 0;
        nbMaterials = 0;
        nbLights = 0;
    }

    HOST void uploadObjects(SceneHelper &helper);

    HOST void uploadLights(SceneHelper &helper);

    HOST void uploadMaterials(SceneHelper &helper);

    D_FORCEINLINE float lightPdf(const float3 &origin, const float3 &dir) const { return 0.f; };
};

void initPassParam(OptixContext &context, OptixLaunchParamsManager<LaunchRadianceParams> &launchParamsManagerRadiance,
                   OptixLaunchParamsManager<LaunchShadowParams> &launchParamsManagerShadow,
                   OptixPipelineManager &pipelineManagerRadiance, OptixPipelineManager &pipelineManagerShadow,
                   OptixSBTManager &sbtManagerRadiance, OptixSBTManager &sbtManagerShadow,
                   Scene &scene, SceneHelper &helper);

void sortLights(SceneHelper &helper);

Scene spheres(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass,
              OptixPassData<LaunchShadowParams> &shadow_pass, float &global_size, int rngmanip = 0);

Scene loadScene(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass,
                OptixPassData<LaunchShadowParams> &shadow_pass, float &global_size, int rngmanip = 0);

void addGround(SceneHelper &helper);

void createMaterials(SceneHelper &helper, int rngmanip = 0);