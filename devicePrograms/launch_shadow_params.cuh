#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/objects/triangle_mesh.cuh"
#include "../src/renderingGPU/materials/material.cuh"

struct LaunchShadowParams
{
    float3* origins = nullptr;
    float3* directions = nullptr;

    int activeCount = 0;

    OptixTraversableHandle traversable = 0;
    
    MeshInstance* meshInstances = nullptr;
    int nbMeshInstances = 0;
    Material* materials = nullptr;
    int nbMaterials = 0;

    float* maxDistances = nullptr;
    int* pixelIndices = nullptr;
    
    float3* transmittance = nullptr;
};