#pragma once

#include "../optix/optix_context.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"

#include <optix_stack_size.h>
#include <string>

inline void initOptix(OptixContext &context, OptixProgramGroupManager &programGroupManager,
                      OptixPipelineManager &pipelineManager, OptixLaunchParamsManager &launchParamsManager,
                      const std::string &raygenPath, const std::string &missPath, const std::string &closestPath)
{
    context.initialize();

    OptixModuleManager raygenModuleManager;

    raygenModuleManager.createFromPath(context.deviceContext, raygenPath);

    std::cout << "Raygen module created successfully" << std::endl;

    OptixModuleManager missModuleManager;
    missModuleManager.createFromPath(context.deviceContext, missPath);

    std::cout << "Miss module created successfully" << std::endl;

    OptixModuleManager chitModuleManager;
    chitModuleManager.createFromPath(context.deviceContext, closestPath);

    std::cout << "Closest module created successfully" << std::endl;

    programGroupManager.create(context.deviceContext, raygenModuleManager.module, missModuleManager.module,
                               chitModuleManager.module);

    std::cout << "RaygenPG = " << programGroupManager.raygenPG << "\nMissPG   = " << programGroupManager.missPG
              << "\nHitPG    = " << programGroupManager.hitPG << std::endl;

    pipelineManager.create(context.deviceContext, raygenModuleManager.getPipelineCompileOptions(),
                           programGroupManager.raygenPG, programGroupManager.missPG, programGroupManager.hitPG);
    std::cout << "Pipeline = " << pipelineManager.pipeline << std::endl;

    launchParamsManager.create();
    std::cout << "d_params = " << launchParamsManager.d_params << std::endl;

    OptixStackSizes stackSizes = {};

    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.raygenPG, &stackSizes, pipelineManager.pipeline));

    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.missPG, &stackSizes, pipelineManager.pipeline));

    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.hitPG, &stackSizes, pipelineManager.pipeline));

    uint32_t dcStackTraversal;
    uint32_t dcStackState;
    uint32_t continuationStack;

    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           1, // maxTraceDepth
                                           0, // maxCCDepth
                                           0, // maxDCDepth
                                           &dcStackTraversal, &dcStackState, &continuationStack));

    OPTIX_CHECK(optixPipelineSetStackSize(pipelineManager.pipeline, dcStackTraversal, dcStackState, continuationStack,
                                          2)); // maxTraversableGraphDepth
}