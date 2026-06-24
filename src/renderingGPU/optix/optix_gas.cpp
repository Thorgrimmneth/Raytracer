#include "optix_gas.h"

#include "../utils/macro.cuh"

void OptixGAS::build(OptixDeviceContext context, CUstream stream, const float3 *d_vertices, uint32_t vertexCount,
                     const uint3 *d_indices, uint32_t triangleCount)
{
    //------------------------------------------------------------------
    // Build input
    //------------------------------------------------------------------

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

    //------------------------------------------------------------------
    // Build options
    //------------------------------------------------------------------

    OptixAccelBuildOptions accelOptions = {};

    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                              OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // OPTIX_BUILD_FLAG_ALLOW_UPDATE si scène dynamique

    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    //------------------------------------------------------------------
    // Memory requirements
    //------------------------------------------------------------------

    OptixAccelBufferSizes gasBufferSizes;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));

    //------------------------------------------------------------------
    // Scratch buffer
    //------------------------------------------------------------------

    CUdeviceptr d_tempBuffer = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));

    //------------------------------------------------------------------
    // GAS output buffer (non compacté)
    //------------------------------------------------------------------

    CUdeviceptr d_uncompactedGas = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_uncompactedGas), gasBufferSizes.outputSizeInBytes));

    //------------------------------------------------------------------
    // Compacted size output
    //------------------------------------------------------------------

    CUdeviceptr d_compactedSize = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compactedSize), sizeof(uint64_t)));

    OptixAccelEmitDesc emitDesc = {};

    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    emitDesc.result = d_compactedSize;

    //------------------------------------------------------------------
    // Build
    //------------------------------------------------------------------

    OptixTraversableHandle uncompactedHandle = 0;

    OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &buildInput, 1, d_tempBuffer,
                                gasBufferSizes.tempSizeInBytes, d_uncompactedGas, gasBufferSizes.outputSizeInBytes,
                                &uncompactedHandle, &emitDesc, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    //------------------------------------------------------------------
    // Read compacted size
    //------------------------------------------------------------------

    uint64_t compactedSize = 0;

    CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void *>(d_compactedSize), sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_compactedSize)));

    //------------------------------------------------------------------
    // Compact if useful
    //------------------------------------------------------------------

    if (compactedSize < gasBufferSizes.outputSizeInBytes)
    {
        CUdeviceptr d_compactedGas = 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compactedGas), compactedSize));

        OptixTraversableHandle compactedHandle = 0;

        OPTIX_CHECK(
            optixAccelCompact(context, stream, uncompactedHandle, d_compactedGas, compactedSize, &compactedHandle));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_uncompactedGas)));

        d_gasBuffer = d_compactedGas;
        handle = compactedHandle;

        std::cout << "GAS compacted: " << gasBufferSizes.outputSizeInBytes / (1024.0 * 1024.0) << " MB -> "
                  << compactedSize / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    else
    {
        d_gasBuffer = d_uncompactedGas;
        handle = uncompactedHandle;

        std::cout << "Compaction not beneficial" << std::endl;
    }

    //------------------------------------------------------------------
    // Cleanup
    //------------------------------------------------------------------

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));

    //------------------------------------------------------------------
    // Stats
    //------------------------------------------------------------------

    std::cout << "GAS handle = " << handle << std::endl;

    std::cout << "Vertices  : " << vertexCount << std::endl;

    std::cout << "Triangles : " << triangleCount << std::endl;
    printf("\n");
}

void OptixGAS::build(OptixContext context, TriangleMesh mesh)
{
    build(context.deviceContext, context.stream, mesh.vertices, mesh.vertexCount, mesh.triangles, mesh.triangleCount);
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