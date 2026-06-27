#include "optix_pipeline_manager.h"

#include <vector>

#include "../utils/simplifiedDef.cuh"

void OptixPipelineManager::create(
    OptixDeviceContext context,
    const OptixPipelineCompileOptions& pipelineCompileOptions,
    OptixProgramGroup raygenPG,
    OptixProgramGroup missPG,
    OptixProgramGroup hitPG,
    OptixProgramGroup anyHitPG
)
{
    std::vector<OptixProgramGroup> groups =
    {
        raygenPG,
        missPG,
        hitPG
    };

    if(anyHitPG != nullptr)
        groups.push_back(anyHitPG);

    OptixPipelineLinkOptions linkOptions = {};

    linkOptions.maxTraceDepth = 1;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(
        optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &linkOptions,
            groups.data(),
            static_cast<unsigned int>(groups.size()),
            log,
            &logSize,
            &pipeline
        )
    );

    if(logSize > 1)
    {
        std::cout
            << "Pipeline log:\n"
            << log
            << std::endl;
    }
}

void OptixPipelineManager::create(
    OptixDeviceContext context,
    const OptixPipelineCompileOptions& pipelineCompileOptions,
    const OptixProgramGroupManager& programGroups
)
{
    std::vector<OptixProgramGroup> groups =
    {
        programGroups.raygenPG,
        programGroups.missPG,
        programGroups.hitPG
    };

    OptixPipelineLinkOptions linkOptions = {};

    linkOptions.maxTraceDepth = 1;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(
        optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &linkOptions,
            groups.data(),
            static_cast<unsigned int>(groups.size()),
            log,
            &logSize,
            &pipeline
        )
    );

    if(logSize > 1)
    {
        std::cout
            << "Pipeline log:\n"
            << log
            << std::endl;
    }
}

void OptixPipelineManager::destroy()
{
    if(pipeline)
    {
        OPTIX_CHECK(
            optixPipelineDestroy(
                pipeline
            )
        );

        pipeline = nullptr;
    }
}