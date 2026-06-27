#pragma once

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <vector_types.h>

#include "../objects/triangle_mesh.cuh"
#include "optix_context.h"

class OptixGAS
{
  public:
    void build(OptixDeviceContext context, CUstream stream, const float3 *d_vertices, uint32_t vertexCount,
               const uint3 *d_indices, uint32_t triangleCount);

    void build(OptixContext context, MeshGeometry mesh);
    void destroy();

    OptixTraversableHandle handle = 0;
    CUdeviceptr d_gasBuffer = 0;
};