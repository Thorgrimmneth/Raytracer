#pragma once

#include "../scene/scene.cuh"
#include "rng.cuh"

struct RayQueue
{
    float3 *origins = nullptr;
    float3 *directions = nullptr;
    float3 *throughputs = nullptr;

    bool *lastBounceWasDelta = nullptr;
    float *lastBsdfPdf = nullptr;
    bool *isInside = nullptr;

    RNG *rng = nullptr;
    int *pixelIndices = nullptr;

    int *activeCount = nullptr;

    void destroy()
    {
        cudaFree(origins);
        cudaFree(directions);
        cudaFree(throughputs);

        cudaFree(lastBounceWasDelta);
        cudaFree(lastBsdfPdf);
        cudaFree(isInside);

        cudaFree(rng);
        cudaFree(pixelIndices);

        cudaFree(activeCount);
    }

    void init(int maxSize)
    {
        cudaMalloc(&origins, maxSize * sizeof(float3));
        cudaMalloc(&directions, maxSize * sizeof(float3));
        cudaMalloc(&throughputs, maxSize * sizeof(float3));

        cudaMalloc(&lastBounceWasDelta, maxSize * sizeof(bool));
        cudaMalloc(&lastBsdfPdf, maxSize * sizeof(float));
        cudaMalloc(&isInside, maxSize * sizeof(bool));

        cudaMalloc(&rng, maxSize * sizeof(RNG));
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));

        cudaMalloc(&activeCount, sizeof(int));
    }
};

struct SortedRayQueue
{
    float3 *origins = nullptr;
    float3 *directions = nullptr;
    float3 *throughputs = nullptr;

    bool *lastBounceWasDelta = nullptr;
    float *lastBsdfPdf = nullptr;
    bool *isInside = nullptr;

    float3 *hitPositions = nullptr;
    float3 *hitNormals = nullptr;
    int *hitMaterialIndices = nullptr;

    RNG *rng = nullptr;
    int *pixelIndices = nullptr;

    void destroy()
    {
        cudaFree(origins);
        cudaFree(directions);
        cudaFree(throughputs);

        cudaFree(lastBounceWasDelta);
        cudaFree(lastBsdfPdf);
        cudaFree(isInside);

        cudaFree(hitPositions);
        cudaFree(hitNormals);
        cudaFree(hitMaterialIndices);

        cudaFree(rng);
        cudaFree(pixelIndices);
    }

    void init(int maxSize)
    {
        cudaMalloc(&origins, maxSize * sizeof(float3));
        cudaMalloc(&directions, maxSize * sizeof(float3));
        cudaMalloc(&throughputs, maxSize * sizeof(float3));

        cudaMalloc(&lastBounceWasDelta, maxSize * sizeof(bool));
        cudaMalloc(&lastBsdfPdf, maxSize * sizeof(float));
        cudaMalloc(&isInside, maxSize * sizeof(bool));

        cudaMalloc(&hitPositions, maxSize * sizeof(float3));
        cudaMalloc(&hitNormals, maxSize * sizeof(float3));
        cudaMalloc(&hitMaterialIndices, maxSize * sizeof(int));

        cudaMalloc(&rng, maxSize * sizeof(RNG));
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
    }
};

struct HitBuffers
{
    float3 *positions = nullptr;
    float3 *normals = nullptr;
    int *materialIndices = nullptr;
    int *mask = nullptr;

    void destroy()
    {
        cudaFree(positions);
        cudaFree(normals);
        cudaFree(materialIndices);
        cudaFree(mask);
    }

    void init(int maxSize)
    {
        cudaMalloc(&positions, maxSize * sizeof(float3));
        cudaMalloc(&normals, maxSize * sizeof(float3));
        cudaMalloc(&materialIndices, maxSize * sizeof(int));
        cudaMalloc(&mask, maxSize * sizeof(int));
    }
};