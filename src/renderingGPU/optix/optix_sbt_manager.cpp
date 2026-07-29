#include "optix_sbt_manager.h"

#include <cuda_runtime.h>

#include "../utils/simplified_def.cuh"

void OptixSBTManager::create(const std::vector<MeshGeometry> &meshes, const OptixProgramGroupManager &pgm,
                             const SDFGeometry &sdfGeometry)
{
    destroy();
    if (meshes.empty() && sdfGeometry.sdfCount == 0)
        return;
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
    size_t meshRecordCount = meshes.size();
    size_t sdfRecordCount = sdfGeometry.sdfCount > 0 ? 1 : 0;
    size_t totalRecordCount = meshRecordCount + sdfRecordCount;
    std::vector<HitRecordSBT> hitRecords(totalRecordCount);
    if (!meshes.empty())
    {

        for (size_t i = 0; i < meshes.size(); ++i)
        {
            auto &rad = hitRecords[i];
            const MeshGeometry &mesh = meshes[i];

            OPTIX_CHECK(optixSbtRecordPackHeader(pgm.meshHitPG, &rad));

            rad.data.mesh.vertices = mesh.vertices;
            rad.data.mesh.normals = mesh.normals;
            rad.data.mesh.uvs = mesh.uvs;

            rad.data.mesh.triangles = mesh.triangles;
        }
    }
    if (sdfGeometry.sdfCount)
    {
        HitRecordSBT rec{};

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.sdfHitPG, &rec));

        rec.data.type = HitData::GeometryType::Sdf;

        rec.data.sdf.sdfs = sdfGeometry.sdfs;

        hitRecords[meshRecordCount] = rec;
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
    sbt.hitgroupRecordCount = totalRecordCount;
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