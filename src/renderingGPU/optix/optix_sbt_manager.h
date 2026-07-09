#pragma once

#include <vector>

#include <optix.h>
#include <optix_stubs.h>

#include "../objects/sdf.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../src/renderingGPU/utils/op.cuh"
#include "optixRayType.h"
#include "optix_program_group_manager.h"

template <typename T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenData
{
};

struct MissData
{
};

struct HitData
{
    enum class GeometryType
    {
        TriangleMesh,
        Sdf
    };

    GeometryType type;

    union {
        struct
        {
            float3 *vertices;
            float3 *normals;
            float2 *uvs;
            uint3 *triangles;
        } mesh;

        struct
        {
            SDF *sdfs;
        } sdf;
    };
};

using RaygenRecord = SbtRecord<RaygenData>;
using MissRecord = SbtRecord<MissData>;
using HitRecordSBT = SbtRecord<HitData>;

class OptixSBTManager
{
  public:
    void create(const std::vector<MeshGeometry> &meshes, const OptixProgramGroupManager &pgm,
                const SDFGeometry &sdfGeometry);

    void destroy();

    OptixShaderBindingTable sbt = {};

  private:
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitRecords = 0;
    CUdeviceptr d_anyHitRecords = 0;
};