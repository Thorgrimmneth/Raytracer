#include "optix_ias.h"

void OptixIAS::build(OptixDeviceContext context, const std::vector<OptixInstance> &instances)
{
    //------------------------------------------------------------------
    // Instances
    //------------------------------------------------------------------

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), sizeof(OptixInstance) * instances.size()));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instances), instances.data(),
                          sizeof(OptixInstance) * instances.size(), cudaMemcpyHostToDevice));

    //------------------------------------------------------------------
    // Build input
    //------------------------------------------------------------------

    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

    buildInput.instanceArray.instances = d_instances;

    buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    //------------------------------------------------------------------
    // Build options
    //------------------------------------------------------------------

    OptixAccelBuildOptions accelOptions = {};

    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // OPTIX_BUILD_FLAG_ALLOW_UPDATE si scène dynamique

    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    //------------------------------------------------------------------
    // Memory usage
    //------------------------------------------------------------------

    OptixAccelBufferSizes bufferSizes;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes));

    //------------------------------------------------------------------
    // Scratch
    //------------------------------------------------------------------

    CUdeviceptr d_temp = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), bufferSizes.tempSizeInBytes));

    //------------------------------------------------------------------
    // Uncompacted IAS
    //------------------------------------------------------------------

    CUdeviceptr d_uncompactedIAS = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_uncompactedIAS), bufferSizes.outputSizeInBytes));

    //------------------------------------------------------------------
    // Compacted size query
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

    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_temp, bufferSizes.tempSizeInBytes,
                                d_uncompactedIAS, bufferSizes.outputSizeInBytes, &uncompactedHandle, &emitDesc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    //------------------------------------------------------------------
    // Read compacted size
    //------------------------------------------------------------------

    uint64_t compactedSize = 0;

    CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void *>(d_compactedSize), sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_compactedSize)));

    //------------------------------------------------------------------
    // Compact
    //------------------------------------------------------------------

    if (compactedSize < bufferSizes.outputSizeInBytes)
    {
        CUdeviceptr d_compactedIAS = 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compactedIAS), compactedSize));

        OptixTraversableHandle compactedHandle = 0;

        OPTIX_CHECK(optixAccelCompact(context, 0, uncompactedHandle, d_compactedIAS, compactedSize, &compactedHandle));

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_uncompactedIAS)));

        d_buffer = d_compactedIAS;
        handle = compactedHandle;

        std::cout << "IAS compacted: " << bufferSizes.outputSizeInBytes / (1024.0 * 1024.0) << " MB -> "
                  << compactedSize / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    else
    {
        d_buffer = d_uncompactedIAS;
        handle = uncompactedHandle;
    }

    //------------------------------------------------------------------
    // Cleanup
    //------------------------------------------------------------------

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
}