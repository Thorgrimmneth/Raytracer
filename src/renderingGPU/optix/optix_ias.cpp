#include "optix_ias.h"

void OptixIAS::build(OptixDeviceContext context, const std::vector<OptixInstance> &instances)
{

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), sizeof(OptixInstance) * instances.size()));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instances), instances.data(),
                          sizeof(OptixInstance) * instances.size(), cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

    buildInput.instanceArray.instances = d_instances;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions accelOptions = {};

    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;

    optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes);

    CUdeviceptr d_temp;

    cudaMalloc(reinterpret_cast<void **>(&d_temp), bufferSizes.tempSizeInBytes);

    cudaMalloc(reinterpret_cast<void **>(&d_buffer), bufferSizes.outputSizeInBytes);

    optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_temp, bufferSizes.tempSizeInBytes, d_buffer,
                    bufferSizes.outputSizeInBytes, &handle, nullptr, 0);
}