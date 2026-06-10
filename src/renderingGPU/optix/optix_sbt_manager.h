#pragma once

#include <optix.h>

#include "optix_sbt_manager.h"
#include "optix_program_group_manager.h"

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
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
};

class OptixSBTManager
{
  public:
    void create(OptixProgramGroup raygenPG, OptixProgramGroup missPG, OptixProgramGroup hitPG);
    void create(const OptixProgramGroupManager& programGroups)
    {
        create(programGroups.raygenPG, programGroups.missPG, programGroups.hitPG);
    }

    void destroy();

    OptixShaderBindingTable sbt = {};

  private:
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitRecord = 0;
};