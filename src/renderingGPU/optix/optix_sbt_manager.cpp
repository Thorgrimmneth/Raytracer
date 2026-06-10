#include "optix_sbt_manager.h"

#include "optix_sbt_manager.h"

void OptixSBTManager::create(OptixProgramGroup raygenPG, OptixProgramGroup missPG, OptixProgramGroup hitPG)
{
    sbt = {};

    //
    // Raygen
    //

    SbtRecord<RaygenData> raygenRecord = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(SbtRecord<RaygenData>)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), &raygenRecord, sizeof(SbtRecord<RaygenData>),
                          cudaMemcpyHostToDevice));

    sbt.raygenRecord = d_raygenRecord;

    //
    // Miss
    //

    SbtRecord<MissData> missRecord = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &missRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_missRecord), sizeof(SbtRecord<MissData>)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_missRecord), &missRecord, sizeof(SbtRecord<MissData>),
                          cudaMemcpyHostToDevice));

    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
    sbt.missRecordCount = 1;

    //
    // Hitgroup
    //

    SbtRecord<HitData> hitRecord = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG, &hitRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitRecord), sizeof(SbtRecord<HitData>)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitRecord), &hitRecord, sizeof(SbtRecord<HitData>),
                          cudaMemcpyHostToDevice));

    sbt.hitgroupRecordBase = d_hitRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
    sbt.hitgroupRecordCount = 1;
}

void OptixSBTManager::destroy()
{
    if (d_raygenRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_raygenRecord)));

        d_raygenRecord = 0;
    }

    if (d_missRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_missRecord)));

        d_missRecord = 0;
    }

    if (d_hitRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_hitRecord)));

        d_hitRecord = 0;
    }
}