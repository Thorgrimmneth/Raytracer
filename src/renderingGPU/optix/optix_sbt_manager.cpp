#include "optix_sbt_manager.h"

#include <cuda_runtime.h>

#include "../utils/simplified_def.cuh"
#include "optix_ray_type.h"

void OptixSBTManager::create(const std::vector<MeshGeometry> &meshes, const OptixProgramGroupManager &pgm,
                             const SDFGeometry &sdfGeometry)
{
    destroy();

    if (meshes.empty() && sdfGeometry.sdfCount == 0)
        return;

    // ============================================================
    // RAYGEN
    // ============================================================

    RaygenRecord raygenRecord{};

    OPTIX_CHECK(optixSbtRecordPackHeader(pgm.raygenPG, &raygenRecord));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(RaygenRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), &raygenRecord, sizeof(RaygenRecord),
                          cudaMemcpyHostToDevice));

    // ============================================================
    // MISS
    //
    // [0] = radiance
    // [1] = shadow
    // ============================================================

    const size_t missRecordCount = RAY_TYPE_COUNT;

    std::vector<MissRecord> missRecords(missRecordCount);

    // ------------------------------------------------------------
    // Radiance miss
    // ------------------------------------------------------------

    OPTIX_CHECK(optixSbtRecordPackHeader(pgm.missPGs[RAY_TYPE_RADIANCE], &missRecords[RAY_TYPE_RADIANCE]));

    // ------------------------------------------------------------
    // Shadow miss
    // ------------------------------------------------------------

    OPTIX_CHECK(optixSbtRecordPackHeader(pgm.missPGs[RAY_TYPE_SHADOW], &missRecords[RAY_TYPE_SHADOW]));

    // ------------------------------------------------------------
    // GPU allocation
    // ------------------------------------------------------------

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_missRecord), sizeof(MissRecord) * missRecordCount));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_missRecord), missRecords.data(),
                          sizeof(MissRecord) * missRecordCount, cudaMemcpyHostToDevice));

    // ============================================================
    // HITGROUPS
    // ============================================================

    const size_t meshRecordCount = meshes.size() * RAY_TYPE_COUNT;

    const size_t sdfRecordCount = sdfGeometry.sdfCount > 0 ? RAY_TYPE_COUNT : 0;

    const size_t totalRecordCount = meshRecordCount + sdfRecordCount;

    std::vector<HitRecordSBT> hitRecords(totalRecordCount);

    // ============================================================
    // MESHES
    // ============================================================

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const MeshGeometry &mesh = meshes[i];

        const size_t base = i * RAY_TYPE_COUNT;

        // --------------------------------------------------------
        // RADIANCE
        // --------------------------------------------------------

        HitRecordSBT &radianceRecord = hitRecords[base + RAY_TYPE_RADIANCE];

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.meshHitPGs[RAY_TYPE_RADIANCE], &radianceRecord));

        radianceRecord.data.type = HitData::GeometryType::TriangleMesh;

        radianceRecord.data.mesh.vertices = mesh.vertices;

        radianceRecord.data.mesh.normals = mesh.normals;

        radianceRecord.data.mesh.uvs = mesh.uvs;

        radianceRecord.data.mesh.triangles = mesh.triangles;

        // --------------------------------------------------------
        // SHADOW
        // --------------------------------------------------------

        HitRecordSBT &shadowRecord = hitRecords[base + RAY_TYPE_SHADOW];

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.meshHitPGs[RAY_TYPE_SHADOW], &shadowRecord));

        shadowRecord.data.type = HitData::GeometryType::TriangleMesh;

        shadowRecord.data.mesh.vertices = mesh.vertices;

        shadowRecord.data.mesh.normals = mesh.normals;

        shadowRecord.data.mesh.uvs = mesh.uvs;

        shadowRecord.data.mesh.triangles = mesh.triangles;
    }

    // ============================================================
    // SDF
    //
    // Les SDF sont placés après tous les meshes.
    // ============================================================

    if (sdfGeometry.sdfCount > 0)
    {
        const size_t base = meshRecordCount;

        // --------------------------------------------------------
        // RADIANCE
        // --------------------------------------------------------

        HitRecordSBT &radianceRecord = hitRecords[base + RAY_TYPE_RADIANCE];

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.sdfHitPGs[RAY_TYPE_RADIANCE], &radianceRecord));

        radianceRecord.data.type = HitData::GeometryType::Sdf;

        radianceRecord.data.sdf.sdfs = sdfGeometry.sdfs;

        // --------------------------------------------------------
        // SHADOW
        // --------------------------------------------------------

        HitRecordSBT &shadowRecord = hitRecords[base + RAY_TYPE_SHADOW];

        OPTIX_CHECK(optixSbtRecordPackHeader(pgm.sdfHitPGs[RAY_TYPE_SHADOW], &shadowRecord));

        shadowRecord.data.type = HitData::GeometryType::Sdf;

        shadowRecord.data.sdf.sdfs = sdfGeometry.sdfs;
    }

    // ============================================================
    // HITGROUP GPU ALLOCATION
    // ============================================================

    if (!hitRecords.empty())
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitRecords), sizeof(HitRecordSBT) * hitRecords.size()));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitRecords), hitRecords.data(),
                              sizeof(HitRecordSBT) * hitRecords.size(), cudaMemcpyHostToDevice));
    }

    // ============================================================
    // BUILD SBT
    // ============================================================

    sbt = {};

    // ------------------------------------------------------------
    // Raygen
    // ------------------------------------------------------------

    sbt.raygenRecord = d_raygenRecord;

    // ------------------------------------------------------------
    // Miss
    // ------------------------------------------------------------

    sbt.missRecordBase = d_missRecord;

    sbt.missRecordStrideInBytes = sizeof(MissRecord);

    sbt.missRecordCount = static_cast<unsigned int>(missRecordCount);

    // ------------------------------------------------------------
    // Hitgroups
    // ------------------------------------------------------------

    sbt.hitgroupRecordBase = d_hitRecords;

    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecordSBT);

    sbt.hitgroupRecordCount = static_cast<unsigned int>(totalRecordCount);
}

void OptixSBTManager::destroy()
{
    if (d_raygenRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_raygenRecord)));
    }

    if (d_missRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_missRecord)));
    }

    if (d_hitRecords)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_hitRecords)));
    }

    d_raygenRecord = 0;
    d_missRecord = 0;
    d_hitRecords = 0;

    sbt = {};
}