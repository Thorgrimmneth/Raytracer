#pragma once

#include "../optix/optix_context.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"
#include "../../devicePrograms/optix_launch_params_manager.h"
#include <optional>
#include <optix_stack_size.h>
#include <string>

template<typename LaunchParamsT>
inline void initOptix(OptixContext &context, OptixProgramGroupManager &programGroupManager,
                      OptixPipelineManager &pipelineManager, OptixLaunchParamsManager<LaunchParamsT> &launchParamsManager)
{
    //
    // Pipeline
    //

    pipelineManager.create(context.deviceContext, context.pipelineCompileOptions, programGroupManager);

    //
    // Launch params
    //

    launchParamsManager.create();

    //
    // Stack size
    //

    OptixStackSizes stackSizes = {};

    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.raygenPG, &stackSizes, pipelineManager.pipeline));

    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.missPG, &stackSizes, pipelineManager.pipeline));

    if (programGroupManager.meshHitPG)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.meshHitPG, &stackSizes, pipelineManager.pipeline));
    }

    if (programGroupManager.sdfHitPG)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.sdfHitPG, &stackSizes, pipelineManager.pipeline));
    }

    uint32_t dcStackTraversal = 0;
    uint32_t dcStackState = 0;
    uint32_t continuationStack = 0;

    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0, &dcStackTraversal, &dcStackState, &continuationStack));

    OPTIX_CHECK(
        optixPipelineSetStackSize(pipelineManager.pipeline, dcStackTraversal, dcStackState, continuationStack, 2));
}