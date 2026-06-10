#include "optix_gas.h"

#include "../utils/macro.cuh"

void OptixGAS::build(OptixDeviceContext context, CUstream stream, const float3 *d_vertices, uint32_t vertexCount,
                     const uint3 *d_indices, uint32_t triangleCount)
{
    //
    // Build input
    //

    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr vertexBuffers[] = {reinterpret_cast<CUdeviceptr>(d_vertices)};

    buildInput.triangleArray.vertexBuffers = vertexBuffers;

    buildInput.triangleArray.numVertices = vertexCount;

    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);

    buildInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(d_indices);

    buildInput.triangleArray.numIndexTriplets = triangleCount;

    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);

    uint32_t flags[] = {OPTIX_GEOMETRY_FLAG_NONE};

    buildInput.triangleArray.flags = flags;

    buildInput.triangleArray.numSbtRecords = 1;

    //
    // Build options
    //

    OptixAccelBuildOptions accelOptions = {};

    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    //
    // Required memory
    //

    OptixAccelBufferSizes gasBufferSizes;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));

    //
    // Scratch
    //

    CUdeviceptr d_tempBuffer = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));

    //
    // GAS output
    //

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gasBuffer), gasBufferSizes.outputSizeInBytes));

    //
    // Build
    //

    OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &buildInput, 1, d_tempBuffer,
                                gasBufferSizes.tempSizeInBytes, d_gasBuffer, gasBufferSizes.outputSizeInBytes, &handle,
                                nullptr, 0));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
}

void OptixGAS::destroy()
{
    if (d_gasBuffer)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gasBuffer)));

        d_gasBuffer = 0;
    }

    handle = 0;
}