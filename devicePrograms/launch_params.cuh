#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "../src/renderingGPU/utils/op.cuh"

// Minimal hit record for internal scene queries (lightPdf, etc)
struct OptixHit
{
    float4 position;
    float4 normal;
    float t;
    int materialIndex;
    int objectIndex;
    HitObjectType objectType;
};

struct LaunchParams
{
    float3* origins = nullptr;
    float4* directions = nullptr;

    // Hit data in SoA format (only relevant fields for active kernels)
    float4* hitPositions = nullptr;
    float4* hitNormals = nullptr;
    int* hitMaterialIndices = nullptr;
    
    int* hitMask = nullptr;

    int* activeQueue = nullptr;

    int activeCount = 0;

    OptixTraversableHandle traversable = 0;
};