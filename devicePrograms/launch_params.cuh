#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/objects/triangle_mesh.cuh"

// Minimal hit record for internal scene queries (lightPdf, etc)
struct OptixHit
{
    float3 position;
    float3 normal;
    float t;
    int materialIndex;
    int objectIndex;
    HitObjectType objectType;
};

struct LaunchParams
{
    float3* origins = nullptr;
    float3* directions = nullptr;

    // Hit data in SoA format (only relevant fields for active kernels)
    float3* hitPositions = nullptr;
    float3* hitNormals = nullptr;
    int* hitMaterialIndices = nullptr;
    
    int* hitMask = nullptr;

    int* activeQueue = nullptr;

    int activeCount = 0;

    OptixTraversableHandle traversable = 0;
    
    // Mesh instances for per-instance data lookup (material, geometry index, etc.)
    MeshInstance* meshInstances = nullptr;
    int nbMeshInstances = 0;
};