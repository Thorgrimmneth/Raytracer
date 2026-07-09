#include "optix_module_manager.h"

#include <iostream>



#include "../utils/simplifiedDef.cuh"
#include "../utils/readPTX.h"

void OptixModuleManager::create(OptixContext context, const std::string &ptx)
{
    OptixModuleCompileOptions moduleOptions = {};

    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(context.deviceContext, &moduleOptions, &context.pipelineCompileOptions, ptx.c_str(), ptx.size(), log,
                                  &logSize, &module));

    if (logSize > 1)
    {
        std::cout << "Module compilation log :\n" << log << std::endl;
    }
}

void OptixModuleManager::createFromPath(OptixContext context, const std::string &path)
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