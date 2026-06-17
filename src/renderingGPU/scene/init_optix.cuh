#pragma once

#include "../optix/optix_context.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"

#include <string>

inline void initOptix(OptixContext &context, OptixProgramGroupManager &programGroupManager,
                      OptixPipelineManager &pipelineManager,
                      OptixLaunchParamsManager &launchParamsManager, const std::string &raygenPath,
                      const std::string &missPath, const std::string &closestPath)
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

    OPTIX_CHECK(optixPipelineSetStackSize(pipelineManager.pipeline,
                                          2 * 1024, // directCallableStackSizeFromTraversal
                                          2 * 1024, // directCallableStackSizeFromState
                                          2 * 1024, // continuationStackSize
                                          2         // maxTraversableGraphDepth
                                          ));
}