#include "optix_launch_params_manager.h"

#include "../src/renderingGPU/utils/macro.cuh"

void OptixLaunchParamsManager::create(uint32_t width, uint32_t height)
{
    params.width = width;
    params.height = height;

    params.traversable = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&framebuffer), width * height * sizeof(uchar4)));

    params.framebuffer = framebuffer;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(LaunchParams)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
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

std::vector<uchar4> OptixLaunchParamsManager::downloadFramebuffer() const
{
    std::vector<uchar4> host(params.width * params.height);

    CUDA_CHECK(cudaMemcpy(host.data(), params.framebuffer, host.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));

    return host;
}