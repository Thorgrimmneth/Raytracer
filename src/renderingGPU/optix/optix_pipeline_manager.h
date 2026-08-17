#pragma once

#include <optix.h>
#include "optix_program_group_manager.h"

#include <vector>

class OptixPipelineManager
{
public:

    void create(
        OptixDeviceContext context,
        const OptixPipelineCompileOptions& pipelineCompileOptions,
        const OptixProgramGroupManager &programGroups
    );

    void destroy();

    OptixPipeline pipeline = nullptr;
};