#include "optix_sbt_manager.h"

#include <cuda_runtime.h>

#include "../utils/macro.cuh"

void OptixSBTManager::create(OptixProgramGroup raygenPG, OptixProgramGroup missPG, OptixProgramGroup hitPG,
                             float3 *vertices, float3 *normals, float2 *uvs, uint3 *triangles, int materialIndex)
{
    RaygenRecord rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rg));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(RaygenRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), &rg, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    MissRecord ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &ms));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_missRecord), sizeof(MissRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_missRecord), &ms, sizeof(MissRecord), cudaMemcpyHostToDevice));

    HitRecordSBT hg = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG, &hg));

    hg.data.vertices = vertices;
    hg.data.normals = normals;
    hg.data.uvs = uvs;
    hg.data.triangles = triangles;
    hg.data.materialIndex = materialIndex;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitRecord), sizeof(HitRecordSBT)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitRecord), &hg, sizeof(HitRecordSBT), cudaMemcpyHostToDevice));

    sbt.raygenRecord = d_raygenRecord;

    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = d_hitRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecordSBT);
    sbt.hitgroupRecordCount = 1;
}

void OptixSBTManager::destroy()
{
    if (d_raygenRecord)
        cudaFree(reinterpret_cast<void *>(d_raygenRecord));

    if (d_missRecord)
        cudaFree(reinterpret_cast<void *>(d_missRecord));

    if (d_hitRecord)
        cudaFree(reinterpret_cast<void *>(d_hitRecord));

    d_raygenRecord = 0;
    d_missRecord = 0;
    d_hitRecord = 0;
}