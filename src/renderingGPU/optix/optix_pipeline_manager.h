#pragma once

#include <optix.h>
#include "optix_program_group_manager.h"

class OptixPipelineManager
{
public:

    void create(
        OptixDeviceContext context,
        const OptixPipelineCompileOptions& pipelineCompileOptions,
        OptixProgramGroup raygenPG,
        OptixProgramGroup missPG,
        OptixProgramGroup hitPG,
        OptixProgramGroup anyHitPG = nullptr
    );

    void create(
        OptixDeviceContext context,
        const OptixPipelineCompileOptions& pipelineCompileOptions,
        const OptixProgramGroupManager& programGroups
    );

    void destroy();

    OptixPipeline pipeline = nullptr;
};