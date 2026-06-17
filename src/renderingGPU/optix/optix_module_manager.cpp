#include "optix_module_manager.h"

#include <iostream>

#include <optix.h>

#include "../utils/macro.cuh"
#include "../utils/readPTX.h"

void OptixModuleManager::create(OptixDeviceContext context, const std::string &ptx)
{
    OptixModuleCompileOptions moduleOptions = {};

    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pipelineCompileOptions = {};

    pipelineCompileOptions.usesMotionBlur = false;

    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

    pipelineCompileOptions.numPayloadValues = 2;

    pipelineCompileOptions.numAttributeValues = 2;

    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(context, &moduleOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), log,
                                  &logSize, &module));

    if (logSize > 1)
    {
        std::cout << "Module compilation log :\n" << log << std::endl;
    }
}

void OptixModuleManager::createFromPath(OptixDeviceContext context, const std::string &path)
{
    std::string ptx = loadTextFile(path);
    create(context, ptx);
}

void OptixModuleManager::destroy()
{
    if (module)
    {
        OPTIX_CHECK(optixModuleDestroy(module));

        module = nullptr;
    }
}