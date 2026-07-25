#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/materials/material.cuh"
#include "../renderingGPU/raytracingUtils/ray.cuh"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/objects/triangle_mesh.cuh"
#include "../renderingGPU/utils/shading_data.cuh"
// Minimal hit record for internal scene queries (lightPdf, etc)

struct LaunchRadianceParams
{
    float3* origins = nullptr;
    float3* directions = nullptr;

    HitBuffers hit_buffers;

    int active_count = 0;

    OptixTraversableHandle traversable = 0;
    
    // Mesh instances for per-instance data lookup (material, geometry index, etc.)
    MeshInstance* meshInstances = nullptr;
    Material* materials = nullptr;
    int nbMeshInstances = 0;
    int nbMaterials = 0;
};