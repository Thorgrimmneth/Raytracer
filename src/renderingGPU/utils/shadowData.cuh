#pragma once

struct ShadowRay {
    float3 *origins;
    float3 *directions;
    float3 *contributions;
    int *pixelIndices;
    float *maxDistance;

    void destroy()
    {
        cudaFree(origins);
        cudaFree(directions);
        cudaFree(contributions);
        cudaFree(pixelIndices);
        cudaFree(maxDistance);
    }

    void init(int maxSize)
    {
        cudaMalloc(&origins, maxSize * sizeof(float3));
        cudaMalloc(&directions, maxSize * sizeof(float3));
        cudaMalloc(&contributions, maxSize * sizeof(float3));
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
        cudaMalloc(&maxDistance, maxSize * sizeof(float));
    }
};