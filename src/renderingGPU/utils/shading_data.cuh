#pragma once

#include "../scene/scene.cuh"
#include "rng.cuh"
#include "check.cuh"

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
        if(origins){
            CUDA_CHECK(cudaFree(origins));
            origins = nullptr;
        }
        if(directions){
            CUDA_CHECK(cudaFree(directions));
            directions = nullptr;
        }
        if(throughputs){
            CUDA_CHECK(cudaFree(throughputs));
            throughputs = nullptr;
        }
        if(lastBounceWasDelta){
            CUDA_CHECK(cudaFree(lastBounceWasDelta));
            lastBounceWasDelta = nullptr;
        }
        if(lastBsdfPdf){
            CUDA_CHECK(cudaFree(lastBsdfPdf));
            lastBsdfPdf = nullptr;
        }
        if(isInside){
            CUDA_CHECK(cudaFree(isInside));
            isInside = nullptr;
        }
        if(rng){
            CUDA_CHECK(cudaFree(rng));
            rng = nullptr;
        }
        if(pixelIndices){
            CUDA_CHECK(cudaFree(pixelIndices));
            pixelIndices = nullptr;
        }
        if(activeCount){
            CUDA_CHECK(cudaFree(activeCount));
            activeCount = nullptr;
        }
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
    float *hitDistances = nullptr;
    int *hitTypes = nullptr;
    int *hitObjectIndices = nullptr;

    RNG *rng = nullptr;
    int *pixelIndices = nullptr;

    void destroy()
    {
        if(origins){
            CUDA_CHECK(cudaFree(origins));
            origins = nullptr;
        }
        if(directions){
            CUDA_CHECK(cudaFree(directions));
            directions = nullptr;
        }
        if(throughputs){
            CUDA_CHECK(cudaFree(throughputs));
            throughputs = nullptr;
        }
        if(lastBounceWasDelta){
            CUDA_CHECK(cudaFree(lastBounceWasDelta));
            lastBounceWasDelta = nullptr;
        }
        if(lastBsdfPdf){
            CUDA_CHECK(cudaFree(lastBsdfPdf));
            lastBsdfPdf = nullptr;
        }
        if(isInside){
            CUDA_CHECK(cudaFree(isInside));
            isInside = nullptr;
        }
        if(hitPositions){
            CUDA_CHECK(cudaFree(hitPositions));
            hitPositions = nullptr;
        }
        if(hitNormals){
            CUDA_CHECK(cudaFree(hitNormals));
            hitNormals = nullptr;
        }
        if(hitMaterialIndices){
            CUDA_CHECK(cudaFree(hitMaterialIndices));
            hitMaterialIndices = nullptr;
        }
        if(rng){
            CUDA_CHECK(cudaFree(rng));
            rng = nullptr;
        }
        if(pixelIndices){
            CUDA_CHECK(cudaFree(pixelIndices));
            pixelIndices = nullptr;
        }
        if(hitDistances){
            CUDA_CHECK(cudaFree(hitDistances));
            hitDistances = nullptr;
        }
        if(hitTypes){
            CUDA_CHECK(cudaFree(hitTypes));
            hitTypes = nullptr;
        }
        if(hitObjectIndices){
            CUDA_CHECK(cudaFree(hitObjectIndices));
            hitObjectIndices = nullptr;
        }
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
        cudaMalloc(&hitDistances, maxSize * sizeof(float));
        cudaMalloc(&hitTypes, maxSize * sizeof(int));
        cudaMalloc(&hitObjectIndices, maxSize * sizeof(int));

        cudaMalloc(&rng, maxSize * sizeof(RNG));
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
    }
};

struct HitBuffers
{
    float3 *positions = nullptr;
    float3 *normals = nullptr;
    int *materialIndices = nullptr;
    float *distances = nullptr;
    int *types = nullptr;
    int *objectIndices = nullptr;
    int *mask = nullptr;

    void destroy()
    {
        if(positions)
        {
            CUDA_CHECK(cudaFree(positions));
            positions = nullptr;
        }
        if(normals)
        {
            CUDA_CHECK(cudaFree(normals));
            normals = nullptr;
        }
        if(materialIndices)
        {
            CUDA_CHECK(cudaFree(materialIndices));
            materialIndices = nullptr;
        }
        if(mask)
        {
            CUDA_CHECK(cudaFree(mask));
            mask = nullptr;
        }
        if(distances)
        {
            CUDA_CHECK(cudaFree(distances));
            distances = nullptr;
        }
        if(objectIndices)
        {
            CUDA_CHECK(cudaFree(objectIndices));
            objectIndices = nullptr;
        }
        if(types)
        {
            CUDA_CHECK(cudaFree(types));
            types = nullptr;
        }
    }

    void init(int maxSize)
    {
        cudaMalloc(&positions, maxSize * sizeof(float3));
        cudaMalloc(&normals, maxSize * sizeof(float3));
        cudaMalloc(&materialIndices, maxSize * sizeof(int));
        cudaMalloc(&mask, maxSize * sizeof(int));
        cudaMalloc(&distances, maxSize * sizeof(float));
        cudaMalloc(&objectIndices, maxSize * sizeof(int));
        cudaMalloc(&types, maxSize * sizeof(int));
    }
};