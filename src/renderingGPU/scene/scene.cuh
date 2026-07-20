#pragma once

#include "../utils/simplified_def.cuh"
#include "mesh_loader.cuh"

#include <vector>

#include "../lights/light.cuh"
#include "../materials/material.cuh"

#include "../objects/plane.cuh"
#include "../objects/sphere.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../raytracingUtils/ray.cuh"

#include "../../devicePrograms/launch_radiance_params.cuh"
#include "../../devicePrograms/launch_shadow_params.cuh"
#include "../../devicePrograms/optix_launch_params_manager.h"
#include "../optix/optix_context.h"
#include "../optix/optix_gas.h"
#include "../optix/optix_ias.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"
#include "../optix/optix_sbt_manager.h"
#include "../utils/compute_transform.cuh"
#include "init_optix.cuh"
#include "scene_helper.cuh"
#include "sun_helper.h"

struct Light;

template <typename LaunchParamsT> struct OptixPassData
{
    OptixPipeline pipeline = nullptr;

    OptixShaderBindingTable sbt{};

    CUdeviceptr d_params = 0;

    LaunchParamsT params{};

    void destroy()
    {
        if (d_params)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
            d_params = 0;
        }

        sbt = {};

        if (pipeline)
        {
            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            pipeline = nullptr;
        }

        params = LaunchParamsT{};
    }
};

struct CudaScene
{
    Sphere *spheres;
    Plane *planes;
    MeshGeometry *meshGeometries; // Shared geometry data (loaded once per file)
    MeshInstance *meshInstances;  // Per-instance data (transform, material)
    Material *materials;
    Light *lights;
    float *lightProbabilities;
    float *lightCumulativeWeights;
    SDFGeometry sdfGeometries;
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

    inline HOST void destroy()
    {
        std::cout << "Destroying CudaScene..." << std::endl;
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

        std::cout << "Destroyed buffers" << std::endl;

        radiancePass.destroy();
        std::cout << "Destroyed radiance pass" << std::endl;
        shadowPass.destroy();
        std::cout << "Destroyed shadow pass" << std::endl;

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

    HOST void uploadObjects(CudaSceneHelper &helper);

    HOST void uploadLights(CudaSceneHelper &helper);

    HOST void uploadMaterials(CudaSceneHelper &helper);


    D_FORCEINLINE float lightPdf(const float3 &origin, const float3 &dir) const
    {

        return 0.f;
    };
};

void sortLights(CudaSceneHelper &helper);
CudaScene singleObject(float4 sunDir, int rngmanip = 0);

void addGround(CudaSceneHelper &helper);