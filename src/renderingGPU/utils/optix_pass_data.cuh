#pragma once

#include "../../devicePrograms/launch_radiance_params.cuh"
#include "../../devicePrograms/optix_launch_params_manager.h"
#include <optix_stubs.h>

template <typename LaunchParamsT> struct OptixPassData
{
    OptixPipeline pipeline = nullptr;

    OptixShaderBindingTable sbt{};

    CUdeviceptr d_params = 0;

    LaunchParamsT params{};

    void destroy()
    {
        if (d_params)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
            d_params = 0;
        }

        sbt = {};

        if (pipeline)
        {
            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            pipeline = nullptr;
        }

        params = LaunchParamsT{};
    }
};