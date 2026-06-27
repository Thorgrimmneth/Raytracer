#include "optix_launch_params_manager.h"

#include "../src/renderingGPU/utils/check.cuh"

void OptixLaunchParamsManager::create()
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(LaunchParams)));
}

void OptixLaunchParamsManager::destroy()
{
    if (framebuffer)
    {
        CUDA_CHECK(cudaFree(framebuffer));

        framebuffer = nullptr;
    }

    if (d_params)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));

        d_params = 0;
    }
}