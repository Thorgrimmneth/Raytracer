#pragma once

#include "../../devicePrograms/optix_launch_params_manager.h"
#include "../optix/optix_context.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"
#include <optional>
#include <optix_stack_size.h>
#include <string>

template <typename LaunchParamsT>
inline void initOptix(OptixContext &context, OptixProgramGroupManager &programGroupManager,
                      OptixPipelineManager &pipelineManager,
                      OptixLaunchParamsManager<LaunchParamsT> &launchParamsManager)
{
    // --------------------------------------------------
    // Pipeline
    // --------------------------------------------------

    pipelineManager.create(context.deviceContext, context.pipelineCompileOptions, programGroupManager);

    // --------------------------------------------------
    // Launch params
    // --------------------------------------------------

    launchParamsManager.create();

    // --------------------------------------------------
    // Stack size
    // --------------------------------------------------

    OptixStackSizes stackSizes = {};

    // Raygen
    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.raygenPG, &stackSizes, pipelineManager.pipeline));

    // Miss
    for (OptixProgramGroup pg : programGroupManager.missPGs)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, pipelineManager.pipeline));
    }

    // Mesh hitgroups
    for (OptixProgramGroup pg : programGroupManager.meshHitPGs)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, pipelineManager.pipeline));
    }

    // SDF hitgroups
    for (OptixProgramGroup pg : programGroupManager.sdfHitPGs)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, pipelineManager.pipeline));
    }

    uint32_t dcStackTraversal = 0;
    uint32_t dcStackState = 0;
    uint32_t continuationStack = 0;

    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0, &dcStackTraversal, &dcStackState, &continuationStack));

    OPTIX_CHECK(
        optixPipelineSetStackSize(pipelineManager.pipeline, dcStackTraversal, dcStackState, continuationStack, 2));
}