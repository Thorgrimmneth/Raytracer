#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/raytracingUtils/ray.cuh"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/objects/triangle_mesh.cuh"
#include "../renderingGPU/materials/material.cuh"

// Minimal hit record for internal scene queries (lightPdf, etc)
struct OptixHit
{
    float3 position;
    float3 normal;
    float t;
    int materialIndex;
    int objectIndex;
    Hitobject_type object_type;
};

struct LaunchRadianceParams
{
    float3* origins = nullptr;
    float3* directions = nullptr;

    // Hit data in SoA format (only relevant fields for active kernels)
    float3* hitPositions = nullptr;
    float3* hitNormals = nullptr;
    int* hitMaterialIndices = nullptr;
    
    int* hitMask = nullptr;

    int activeCount = 0;

    OptixTraversableHandle traversable = 0;
    
    // Mesh instances for per-instance data lookup (material, geometry index, etc.)
    MeshInstance* meshInstances = nullptr;
    int nbMeshInstances = 0;
};