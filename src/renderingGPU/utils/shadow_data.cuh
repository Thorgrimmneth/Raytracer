#pragma once

#include "check.cuh"

struct ShadowRayQueue {
    float3 *origins;
    float3 *directions;
    float3 *contributions;
    int *pixelIndices;
    float *maxDistances;
    float3 *transmittance;

    void destroy()
    {
        if(origins)
        {
            CUDA_CHECK(cudaFree(origins));
            origins = nullptr;
        }
        if(directions)
        {
            CUDA_CHECK(cudaFree(directions));
            directions = nullptr;
        }
        if(contributions)
        {
            CUDA_CHECK(cudaFree(contributions));
            contributions = nullptr;
        }
        if(pixelIndices)
        {
            CUDA_CHECK(cudaFree(pixelIndices));
            pixelIndices = nullptr;
        }
        if(maxDistances)
        {
            CUDA_CHECK(cudaFree(maxDistances));
            maxDistances = nullptr;
        }
        if(transmittance)
        {
            CUDA_CHECK(cudaFree(transmittance));
            transmittance = nullptr;
        }
    }

    void init(int maxSize, float &global_size)
    {
        cudaMalloc(&origins, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&directions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&contributions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&maxDistances, maxSize * sizeof(float));
        global_size += maxSize * sizeof(float);
        cudaMalloc(&transmittance, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
    }
};