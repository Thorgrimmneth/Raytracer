#pragma once

#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "launch_radiance_params.cuh"
#include "launch_shadow_params.cuh"
#include "../src/renderingGPU/utils/check.cuh"
#include <vector>

template<typename T>
class OptixLaunchParamsManager
{
public:

    CUdeviceptr d_params = 0;

    T params = {};

    void create()
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_params),
            sizeof(T)));
    }

    void upload()
    {
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_params),
            &params,
            sizeof(T),
            cudaMemcpyHostToDevice));
    }

    void destroy()
    {
        if(d_params)
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

        d_params = 0;
    }
};