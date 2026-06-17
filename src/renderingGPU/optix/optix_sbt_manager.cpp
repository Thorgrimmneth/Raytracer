#include "optix_sbt_manager.h"

#include <cuda_runtime.h>

#include "../utils/macro.cuh"

void OptixSBTManager::create(const OptixProgramGroupManager &pgm, const std::vector<TriangleMesh> &meshes)
{
    destroy();

    //
    // Raygen
    //
    RaygenRecord rg{};
    OPTIX_CHECK(optixSbtRecordPackHeader(pgm.raygenPG, &rg));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(RaygenRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), &rg, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    //
    // Miss
    //
    MissRecord ms{};
    OPTIX_CHECK(optixSbtRecordPackHeader(pgm.missPG, &ms));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_missRecord), sizeof(MissRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_missRecord), &ms, sizeof(MissRecord), cudaMemcpyHostToDevice));

    //
    // Hit groups
    //
    std::vector<HitRecordSBT> hitRecords(meshes.size());

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const TriangleMesh &mesh = meshes[i];

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.hitPG, &hitRecords[i]));

        hitRecords[i].data.vertices = mesh.vertices;
        hitRecords[i].data.normals = mesh.normals;
        hitRecords[i].data.uvs = mesh.uvs;

        hitRecords[i].data.triangles = mesh.triangles;

        hitRecords[i].data.materialIndex = mesh.materialIndex;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitRecords), sizeof(HitRecordSBT) * hitRecords.size()));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitRecords), hitRecords.data(),
                          sizeof(HitRecordSBT) * hitRecords.size(), cudaMemcpyHostToDevice));

    //
    // SBT
    //
    sbt = {};

    sbt.raygenRecord = d_raygenRecord;

    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = d_hitRecords;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecordSBT);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
}

void OptixSBTManager::destroy()
{
    if (d_raygenRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_raygenRecord)));

    if (d_missRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_missRecord)));

    if (d_hitRecords)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_hitRecords)));

    d_raygenRecord = 0;
    d_missRecord = 0;
    d_hitRecords = 0;

    sbt = {};
}