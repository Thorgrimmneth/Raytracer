#pragma once

#include "rng.cuh"
#include "check.cuh"

struct RayQueue
{
    float3 *origins;
    float3 *directions;
    float3 *throughputs;

    bool *lastBounceWasDelta;
    float *lastBsdfPdf;
    bool *isInside;

    RNG *rng;
    int *pixelIndices;

    int *active_count;

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
        if(active_count){
            CUDA_CHECK(cudaFree(active_count));
            active_count = nullptr;
        }
    }

    void init(int maxSize, float &global_size)
    {
        cudaMalloc(&origins, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&directions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&throughputs, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);

        cudaMalloc(&lastBounceWasDelta, maxSize * sizeof(bool));
        global_size += maxSize * sizeof(bool);
        cudaMalloc(&lastBsdfPdf, maxSize * sizeof(float));
        global_size += maxSize * sizeof(float);
        cudaMalloc(&isInside, maxSize * sizeof(bool));
        global_size += maxSize * sizeof(bool);

        cudaMalloc(&rng, maxSize * sizeof(RNG));
        global_size += maxSize * sizeof(RNG);
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
        global_size += maxSize * sizeof(int);

        cudaMalloc(&active_count, sizeof(int));
        global_size += sizeof(int);
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

    void init(int maxSize, float &global_size)
    {
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&origins, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&directions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&throughputs, maxSize * sizeof(float3));

        global_size += maxSize * sizeof(bool);
        cudaMalloc(&lastBounceWasDelta, maxSize * sizeof(bool));
        global_size += maxSize * sizeof(float);
        cudaMalloc(&lastBsdfPdf, maxSize * sizeof(float));
        global_size += maxSize * sizeof(bool);
        cudaMalloc(&isInside, maxSize * sizeof(bool));

        global_size += maxSize * sizeof(float3);
        cudaMalloc(&hitPositions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&hitNormals, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&hitMaterialIndices, maxSize * sizeof(int));
        global_size += maxSize * sizeof(float);
        cudaMalloc(&hitDistances, maxSize * sizeof(float));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&hitTypes, maxSize * sizeof(int));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&hitObjectIndices, maxSize * sizeof(int));

        global_size += maxSize * sizeof(RNG);
        cudaMalloc(&rng, maxSize * sizeof(RNG));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&pixelIndices, maxSize * sizeof(int));
    }
};

struct HitBuffers
{
    float3 *positions;
    float3 *normals;
    int *materialIndices;
    float *distances;
    int *types;
    int *objectIndices;
    int *mask;

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

    void init(int maxSize, float &global_size)
    {
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&positions, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(float3);
        cudaMalloc(&normals, maxSize * sizeof(float3));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&materialIndices, maxSize * sizeof(int));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&mask, maxSize * sizeof(int));
        global_size += maxSize * sizeof(float);
        cudaMalloc(&distances, maxSize * sizeof(float));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&objectIndices, maxSize * sizeof(int));
        global_size += maxSize * sizeof(int);
        cudaMalloc(&types, maxSize * sizeof(int));
    }
};