#pragma once

#include "../optix/optix_context.h"
#include "../optix/optix_module_manager.h"
#include "../optix/optix_pipeline_manager.h"
#include "../optix/optix_program_group_manager.h"

#include <optional>
#include <optix_stack_size.h>
#include <string>

template<typename LaunchParamsT>
inline void initOptix(OptixContext &context, OptixProgramGroupManager &programGroupManager,
                      OptixPipelineManager &pipelineManager, OptixLaunchParamsManager<LaunchParamsT> &launchParamsManager,
                      const std::string &raygenPath, const std::string &raygenName, const std::string &missPath,
                      const std::string &missName, const std::string &closestPath = "",
                      const std::string &closestName = "", const std::string &anyHitPath = "",
                      const std::string &anyHitName = "", const std::string &intersectionPath = "",
                      const std::string &intersectionName = "")
{
    //
    // Modules
    //

    OptixModuleManager raygenModule;
    raygenModule.createFromPath(context.deviceContext, raygenPath);

    OptixModuleManager missModule;
    missModule.createFromPath(context.deviceContext, missPath);

    std::optional<OptixModuleManager> closestModule;
    if (!closestPath.empty())
    {
        closestModule.emplace();
        closestModule->createFromPath(context.deviceContext, closestPath);
    }

    std::optional<OptixModuleManager> anyHitModule;
    if (!anyHitPath.empty())
    {
        anyHitModule.emplace();
        anyHitModule->createFromPath(context.deviceContext, anyHitPath);
    }
    std::optional<OptixModuleManager> intersectionModule;
    if (!intersectionPath.empty())
    {
        intersectionModule.emplace();
        intersectionModule->createFromPath(context.deviceContext, intersectionPath);
    }

    //
    // Program Groups
    //

    programGroupManager.create(
        context.deviceContext, raygenModule.module, raygenName.c_str(), missModule.module, missName.c_str(),
        closestModule ? closestModule->module : nullptr, closestName.empty() ? "" : closestName.c_str(),
        anyHitModule ? anyHitModule->module : nullptr, anyHitName.empty() ? "" : anyHitName.c_str(),
        intersectionModule ? intersectionModule->module : nullptr,
        intersectionName.empty() ? "" : intersectionName.c_str());

    //
    // Pipeline
    //

    pipelineManager.create(context.deviceContext, raygenModule.getPipelineCompileOptions(), programGroupManager);

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

    if (programGroupManager.hitPG)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroupManager.hitPG, &stackSizes, pipelineManager.pipeline));
    }

    uint32_t dcStackTraversal = 0;
    uint32_t dcStackState = 0;
    uint32_t continuationStack = 0;

    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0, &dcStackTraversal, &dcStackState, &continuationStack));

    OPTIX_CHECK(
        optixPipelineSetStackSize(pipelineManager.pipeline, dcStackTraversal, dcStackState, continuationStack, 2));
}